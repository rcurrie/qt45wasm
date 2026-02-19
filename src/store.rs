use anyhow::Result;
use rusqlite::{params, Connection};

use crate::types::{ComputeTier, StoredFunction, TestCase, Value};

/// Persistent storage for functions and their compiled WASM binaries.
pub struct FunctionStore {
    conn: Connection,
}

impl FunctionStore {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                signature TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                source_code TEXT,
                wasm_binary BLOB,
                created_at TEXT DEFAULT (datetime('now')),
                last_used_at TEXT,
                call_count INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                compute_tier TEXT NOT NULL DEFAULT 'scalar',
                shader_source TEXT
            );
            CREATE TABLE IF NOT EXISTS function_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL REFERENCES functions(name),
                input_json TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                passed INTEGER DEFAULT 0
            );

            -- FTS5 external content table for keyword search
            CREATE VIRTUAL TABLE IF NOT EXISTS functions_fts USING fts5(
                name, description, content=functions, content_rowid=id
            );

            -- Sync triggers for FTS5 (INSERT OR REPLACE = DELETE + INSERT)
            CREATE TRIGGER IF NOT EXISTS functions_ai AFTER INSERT ON functions BEGIN
                INSERT INTO functions_fts(rowid, name, description)
                VALUES (new.id, new.name, new.description);
            END;
            CREATE TRIGGER IF NOT EXISTS functions_ad AFTER DELETE ON functions BEGIN
                INSERT INTO functions_fts(functions_fts, rowid, name, description)
                VALUES ('delete', old.id, old.name, old.description);
            END;
            CREATE TRIGGER IF NOT EXISTS functions_au AFTER UPDATE ON functions BEGIN
                INSERT INTO functions_fts(functions_fts, rowid, name, description)
                VALUES ('delete', old.id, old.name, old.description);
                INSERT INTO functions_fts(rowid, name, description)
                VALUES (new.id, new.name, new.description);
            END;",
        )?;

        Ok(Self { conn })
    }

    /// Look up a function by name.
    pub fn get(&self, name: &str) -> Result<Option<StoredFunction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, signature, source_lang, source_code, wasm_binary,
                    call_count, is_verified, compute_tier, shader_source
             FROM functions WHERE name = ?1",
        )?;
        let mut rows = stmt.query(params![name])?;

        if let Some(row) = rows.next()? {
            let tier_str: String = row.get(9)?;
            Ok(Some(StoredFunction {
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
            }))
        } else {
            Ok(None)
        }
    }

    /// Save a new function (or replace an existing one by name).
    pub fn save(
        &self,
        name: &str,
        description: &str,
        signature: &str,
        source_lang: &str,
        source_code: &str,
        wasm_binary: &[u8],
    ) -> Result<()> {
        self.save_with_tier(name, description, signature, source_lang, source_code, wasm_binary, ComputeTier::Scalar, None)
    }

    /// Save a new function with explicit compute tier and optional shader source.
    pub fn save_with_tier(
        &self,
        name: &str,
        description: &str,
        signature: &str,
        source_lang: &str,
        source_code: &str,
        wasm_binary: &[u8],
        tier: ComputeTier,
        shader_source: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO functions
             (name, description, signature, source_lang, source_code, wasm_binary, compute_tier, shader_source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![name, description, signature, source_lang, source_code, wasm_binary, tier.as_str(), shader_source],
        )?;
        println!("  [store] saved function: '{}' (tier: {})", name, tier);
        Ok(())
    }

    /// Increment the call count and update last_used_at.
    pub fn record_call(&self, name: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE functions SET call_count = call_count + 1, last_used_at = datetime('now') WHERE name = ?1",
            params![name],
        )?;
        Ok(())
    }

    /// Save a test case for a function.
    pub fn save_test(&self, function_name: &str, input: &[Value], expected: &[Value]) -> Result<i64> {
        let input_json = serde_json::to_string(
            &input.iter().map(|v| v.to_json()).collect::<Vec<_>>(),
        )?;
        let expected_json = serde_json::to_string(
            &expected.iter().map(|v| v.to_json()).collect::<Vec<_>>(),
        )?;
        self.conn.execute(
            "INSERT INTO function_tests (function_name, input_json, expected_json) VALUES (?1, ?2, ?3)",
            params![function_name, input_json, expected_json],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Get all test cases for a function.
    pub fn get_tests(&self, function_name: &str) -> Result<Vec<TestCase>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, input_json, expected_json FROM function_tests WHERE function_name = ?1",
        )?;
        let rows = stmt.query_map(params![function_name], |row| {
            let id: i64 = row.get(0)?;
            let input_json: String = row.get(1)?;
            let expected_json: String = row.get(2)?;
            Ok((id, input_json, expected_json))
        })?;

        let mut tests = Vec::new();
        for row in rows {
            let (id, input_json, expected_json) = row?;
            let input_vals: Vec<serde_json::Value> = serde_json::from_str(&input_json)?;
            let expected_vals: Vec<serde_json::Value> = serde_json::from_str(&expected_json)?;

            tests.push(TestCase {
                id: Some(id),
                input: input_vals.iter().map(Value::from_json).collect::<Result<Vec<_>>>()?,
                expected: expected_vals.iter().map(Value::from_json).collect::<Result<Vec<_>>>()?,
            });
        }
        Ok(tests)
    }

    /// Mark a test as passed.
    pub fn mark_test_passed(&self, test_id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE function_tests SET passed = 1 WHERE id = ?1",
            params![test_id],
        )?;
        Ok(())
    }

    /// Set the verified flag on a function.
    pub fn set_verified(&self, name: &str, verified: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE functions SET is_verified = ?1 WHERE name = ?2",
            params![verified as i64, name],
        )?;
        Ok(())
    }

    /// Get a reference to the underlying connection (for the search module).
    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Delete a function by name, including its test cases.
    pub fn delete(&self, name: &str) -> Result<()> {
        self.conn.execute(
            "DELETE FROM function_tests WHERE function_name = ?1",
            params![name],
        )?;
        self.conn.execute(
            "DELETE FROM functions WHERE name = ?1",
            params![name],
        )?;
        Ok(())
    }

    /// List all stored functions.
    pub fn list(&self) -> Result<Vec<StoredFunction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, signature, source_lang, source_code, wasm_binary,
                    call_count, is_verified, compute_tier, shader_source
             FROM functions ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
            let tier_str: String = row.get(9)?;
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
                compute_tier: ComputeTier::from_str(&tier_str),
                shader_source: row.get(10)?,
            })
        })?;
        let mut fns = Vec::new();
        for row in rows {
            fns.push(row?);
        }
        Ok(fns)
    }
}
