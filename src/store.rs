use anyhow::Result;
use rusqlite::{params, Connection};

use crate::types::{StoredFunction, TestCase, Value};

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
                is_verified INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS function_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL REFERENCES functions(name),
                input_json TEXT NOT NULL,
                expected_json TEXT NOT NULL,
                passed INTEGER DEFAULT 0
            );",
        )?;
        Ok(Self { conn })
    }

    /// Look up a function by name.
    pub fn get(&self, name: &str) -> Result<Option<StoredFunction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, signature, source_lang, source_code, wasm_binary, call_count, is_verified
             FROM functions WHERE name = ?1",
        )?;
        let mut rows = stmt.query(params![name])?;

        if let Some(row) = rows.next()? {
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
        self.conn.execute(
            "INSERT OR REPLACE INTO functions (name, description, signature, source_lang, source_code, wasm_binary)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![name, description, signature, source_lang, source_code, wasm_binary],
        )?;
        println!("  [store] saved function: '{}'", name);
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
    #[allow(dead_code)]
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

    /// List all stored functions (name, signature, call_count, is_verified).
    pub fn list(&self) -> Result<Vec<StoredFunction>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, signature, source_lang, source_code, wasm_binary, call_count, is_verified
             FROM functions ORDER BY name",
        )?;
        let rows = stmt.query_map([], |row| {
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
        })?;
        let mut fns = Vec::new();
        for row in rows {
            fns.push(row?);
        }
        Ok(fns)
    }
}
