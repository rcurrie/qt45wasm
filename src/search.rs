use anyhow::Result;
use rusqlite::params;
use strsim::jaro_winkler;

use crate::store::FunctionStore;
use crate::types::{ComputeTier, StoredFunction};

const SEARCH_LIMIT: usize = 5;
const NAME_SIMILARITY_THRESHOLD: f64 = 0.75;
const BM25_WEIGHT_NAME: f64 = 10.0;
const BM25_WEIGHT_DESC: f64 = 1.0;

/// A search result with a relevance score.
pub struct SearchResult {
    pub function: StoredFunction,
    pub score: f64,
    pub match_source: String,
}

/// Search combining FTS5 (BM25 keyword ranking) and string similarity on function names.
pub struct FunctionSearch;

impl FunctionSearch {
    pub fn new() -> Self {
        Self
    }

    /// Search for an existing function matching the given request.
    /// Uses FTS5/BM25 for keyword search plus jaro-winkler on function names.
    pub fn find(
        &self,
        store: &FunctionStore,
        name: &str,
        description: &str,
        signature: &str,
    ) -> Result<Option<SearchResult>> {
        let query = format!("{name} {description}");

        // Get FTS5 keyword matches ranked by BM25 (name weighted 10x)
        let fts_results = self.fts_search(store, &query);

        // Get all function names for string similarity matching
        let name_results = self.name_similarity_search(store, name)?;

        // Merge: prefer functions that appear in both, otherwise rank by score
        let mut best: Option<SearchResult> = None;

        // Check FTS results first (keyword relevance)
        for (func, bm25_rank) in &fts_results {
            if func.signature != signature {
                continue;
            }
            // BM25 rank is negative (lower = better), normalize to positive score
            let fts_score = 1.0 / (1.0 + bm25_rank.abs());
            // Boost if name is also similar
            let name_sim = jaro_winkler(&func.name, name);
            let combined = fts_score + if name_sim > NAME_SIMILARITY_THRESHOLD { name_sim } else { 0.0 };

            if best.as_ref().is_none_or(|b| combined > b.score) {
                best = Some(SearchResult {
                    function: func.clone(),
                    score: combined,
                    match_source: format!("fts(bm25={bm25_rank:.2}, name_sim={name_sim:.2})"),
                });
            }
        }

        // Check name similarity results (catches synonyms FTS might miss)
        for (func, similarity) in &name_results {
            if func.signature != signature {
                continue;
            }
            if best.as_ref().is_none_or(|b| *similarity > b.score) {
                best = Some(SearchResult {
                    function: func.clone(),
                    score: *similarity,
                    match_source: format!("name_sim({similarity:.2})"),
                });
            }
        }

        Ok(best)
    }

    /// Full-text search using FTS5 with BM25 ranking (name weighted 10x over description).
    fn fts_search(&self, store: &FunctionStore, query: &str) -> Vec<(StoredFunction, f64)> {
        let sanitized = sanitize_fts_query(query);
        if sanitized.is_empty() {
            return Vec::new();
        }

        let conn = store.connection();
        let result = (|| -> Result<Vec<(StoredFunction, f64)>> {
            let mut stmt = conn.prepare(
                &format!(
                    "SELECT f.id, f.name, f.description, f.signature, f.source_lang, f.source_code,
                            f.wasm_binary, f.call_count, f.is_verified, f.compute_tier, f.shader_source,
                            bm25(functions_fts, {BM25_WEIGHT_NAME}, {BM25_WEIGHT_DESC}) as rank
                     FROM functions_fts
                     JOIN functions f ON f.id = functions_fts.rowid
                     WHERE functions_fts MATCH ?1
                     ORDER BY rank
                     LIMIT ?2"
                ),
            )?;
            let rows: Vec<(StoredFunction, f64)> = stmt
                .query_map(params![sanitized, SEARCH_LIMIT as i64], |row| {
                    let tier_str: String = row.get(9)?;
                    Ok((
                        StoredFunction {
                            id: row.get(0)?,
                            name: row.get(1)?,
                            description: row.get(2)?,
                            signature: row.get(3)?,
                            source_lang: row.get(4)?,
                            source_code: row.get(5)?,
                            wasm_binary: row.get(6)?,
                            call_count: row.get(7)?,
                            is_verified: row.get::<_, i64>(8)? != 0,
                            compute_tier: ComputeTier::from_str(&tier_str),
                            shader_source: row.get(10)?,
                        },
                        row.get::<_, f64>(11)?,
                    ))
                })?
                .collect::<Result<_, _>>()?;
            Ok(rows)
        })();

        result.unwrap_or_default()
    }

    /// Search by function name similarity using jaro-winkler distance.
    fn name_similarity_search(&self, store: &FunctionStore, query_name: &str) -> Result<Vec<(StoredFunction, f64)>> {
        let all = store.list()?;
        let mut matches: Vec<(StoredFunction, f64)> = all
            .into_iter()
            .map(|f| {
                let sim = jaro_winkler(&f.name, query_name);
                (f, sim)
            })
            .filter(|(_, sim)| *sim >= NAME_SIMILARITY_THRESHOLD)
            .collect();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(SEARCH_LIMIT);
        Ok(matches)
    }
}

/// Sanitize a query for FTS5 by quoting each word and joining with OR.
fn sanitize_fts_query(input: &str) -> String {
    input
        .split_whitespace()
        .map(|word| format!("\"{}\"", word.replace('"', "")))
        .collect::<Vec<_>>()
        .join(" OR ")
}
