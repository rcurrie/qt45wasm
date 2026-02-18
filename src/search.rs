use std::collections::HashMap;

use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rusqlite::params;
use zerocopy::IntoBytes;

use crate::store::FunctionStore;
use crate::types::StoredFunction;

const RRF_K: f64 = 60.0;
const SCORE_THRESHOLD: f64 = 0.025;
const MAX_VECTOR_DISTANCE: f64 = 0.16; // cosine distance: 0 = identical, 2 = opposite
const SEARCH_LIMIT: usize = 5;

/// A search result with a fused relevance score.
pub struct SearchResult {
    pub function: StoredFunction,
    pub score: f64,
    pub match_source: String,
}

/// Hybrid search combining FTS5 (keyword) and sqlite-vec (semantic).
pub struct HybridSearch {
    model: TextEmbedding,
}

impl HybridSearch {
    /// Initialize the search system. Downloads the embedding model on first run (~50MB).
    pub fn new(store: &FunctionStore) -> Result<Self> {
        println!("[search] initializing embedding model...");
        let cache_dir = std::env::current_dir()?.join("cache");
        // Override HF_HOME so fastembed/hf-hub won't ignore our cache_dir
        // Safety: called at init before any threads are spawned
        unsafe { std::env::set_var("HF_HOME", &cache_dir) };
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::BGESmallENV15)
                .with_cache_dir(cache_dir)
                .with_show_download_progress(true),
        )?;

        let mut search = Self { model };
        search.backfill_embeddings(store)?;

        Ok(search)
    }

    /// Generate embeddings for functions that are stored but not yet in vec_functions.
    fn backfill_embeddings(&mut self, store: &FunctionStore) -> Result<()> {
        let conn = store.connection();

        let mut stmt = conn.prepare(
            "SELECT f.id, f.name, f.description
             FROM functions f
             LEFT JOIN vec_functions v ON f.id = v.function_id
             WHERE v.function_id IS NULL",
        )?;

        let missing: Vec<(i64, String, String)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<Result<_, _>>()?;

        if missing.is_empty() {
            return Ok(());
        }

        println!("[search] backfilling {} embeddings...", missing.len());
        let texts: Vec<String> = missing
            .iter()
            .map(|(_, name, desc)| format!("{name}: {desc}"))
            .collect();

        let embeddings = self.model.embed(&texts, None)?;

        for ((id, _, _), embedding) in missing.iter().zip(embeddings.iter()) {
            store.upsert_embedding(*id, embedding)?;
        }

        Ok(())
    }

    /// Generate and store an embedding for a newly saved function.
    pub fn embed_function(&mut self, store: &FunctionStore, name: &str, description: &str) -> Result<()> {
        let text = format!("{name}: {description}");
        let embeddings = self.model.embed(&[text], None)?;
        if let (Some(id), Some(embedding)) = (store.get_id(name)?, embeddings.first()) {
            store.upsert_embedding(id, embedding)?;
        }
        Ok(())
    }

    /// Search for an existing function matching the given request.
    /// Returns the best match if it exceeds the score threshold and has a compatible signature.
    pub fn find(
        &mut self,
        store: &FunctionStore,
        name: &str,
        description: &str,
        signature: &str,
    ) -> Result<Option<SearchResult>> {
        let query = format!("{name} {description}");

        let fts_results = self.fts_search(store, &query);
        let vec_results = self.vec_search(store, &query)?;

        // Build a map of vector distances for filtering
        let vec_distances: HashMap<i64, f64> = vec_results.iter().copied().collect();

        let merged = self.rrf_merge(&fts_results, &vec_results);

        // Find the best result with a matching signature and acceptable distance
        for (func_id, score, source) in &merged {
            if *score < SCORE_THRESHOLD {
                break;
            }

            // Require vector distance below threshold (skip if not in vec results)
            let distance = vec_distances.get(func_id).copied().unwrap_or(f64::MAX);
            if distance > MAX_VECTOR_DISTANCE {
                continue;
            }

            let conn = store.connection();
            let func = conn.query_row(
                "SELECT id, name, description, signature, source_lang, source_code, wasm_binary, call_count, is_verified
                 FROM functions WHERE id = ?1",
                params![func_id],
                |row| {
                    Ok(StoredFunction {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        description: row.get(2)?,
                        signature: row.get(3)?,
                        source_lang: row.get(4)?,
                        source_code: row.get(5)?,
                        wasm_binary: row.get(6)?,
                        call_count: row.get(7)?,
                        is_verified: row.get::<_, i64>(8)? != 0,
                    })
                },
            )?;

            if func.signature == signature {
                return Ok(Some(SearchResult {
                    function: func,
                    score: *score,
                    match_source: source.clone(),
                }));
            }
            println!(
                "  [search] found '{}' but signature mismatch: {} vs {}",
                func.name, func.signature, signature
            );
        }

        Ok(None)
    }

    /// Full-text search using FTS5 with BM25 ranking.
    /// Returns (function_id, bm25_rank) pairs. Non-fatal: returns empty on error.
    fn fts_search(&self, store: &FunctionStore, query: &str) -> Vec<(i64, f64)> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Vec::new();
        }

        let conn = store.connection();
        let result = (|| -> Result<Vec<(i64, f64)>> {
            let mut stmt = conn.prepare(
                "SELECT f.id, rank
                 FROM functions_fts
                 JOIN functions f ON f.id = functions_fts.rowid
                 WHERE functions_fts MATCH ?1
                 ORDER BY rank
                 LIMIT ?2",
            )?;
            let rows: Vec<(i64, f64)> = stmt
                .query_map(params![sanitized, SEARCH_LIMIT as i64], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        })();

        result.unwrap_or_default()
    }

    /// Semantic search using sqlite-vec cosine distance.
    fn vec_search(&mut self, store: &FunctionStore, query: &str) -> Result<Vec<(i64, f64)>> {
        let embeddings = self.model.embed(&[query], None)?;
        let query_embedding = embeddings
            .first()
            .ok_or_else(|| anyhow::anyhow!("Failed to generate query embedding"))?;

        let conn = store.connection();
        let mut stmt = conn.prepare(
            "SELECT function_id, distance
             FROM vec_functions
             WHERE embedding MATCH ?1 AND k = ?2",
        )?;

        let results: Vec<(i64, f64)> = stmt
            .query_map(
                params![query_embedding.as_slice().as_bytes(), SEARCH_LIMIT as i64],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
            .collect::<Result<_, _>>()?;

        Ok(results)
    }

    /// Combine FTS5 and vector search results using Reciprocal Rank Fusion.
    fn rrf_merge(
        &self,
        fts_results: &[(i64, f64)],
        vec_results: &[(i64, f64)],
    ) -> Vec<(i64, f64, String)> {
        let mut scores: HashMap<i64, (f64, bool, bool)> = HashMap::new();

        for (rank, (id, _)) in fts_results.iter().enumerate() {
            let rrf = 1.0 / (RRF_K + (rank as f64 + 1.0));
            let entry = scores.entry(*id).or_insert((0.0, false, false));
            entry.0 += rrf;
            entry.1 = true;
        }

        for (rank, (id, _)) in vec_results.iter().enumerate() {
            let rrf = 1.0 / (RRF_K + (rank as f64 + 1.0));
            let entry = scores.entry(*id).or_insert((0.0, false, false));
            entry.0 += rrf;
            entry.2 = true;
        }

        let mut merged: Vec<(i64, f64, String)> = scores
            .into_iter()
            .map(|(id, (score, in_fts, in_vec))| {
                let source = match (in_fts, in_vec) {
                    (true, true) => "both".to_string(),
                    (true, false) => "fts".to_string(),
                    (false, true) => "vec".to_string(),
                    _ => unreachable!(),
                };
                (id, score, source)
            })
            .collect();

        merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged
    }
}

/// Sanitize a query for FTS5 by quoting each word and joining with OR.
/// OR semantics allow partial keyword overlap (e.g., "adds two" matches "adds two integers").
fn sanitize_fts_query(input: &str) -> String {
    input
        .split_whitespace()
        .map(|word| format!("\"{}\"", word.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
}
