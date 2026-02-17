use anyhow::Result;
use rusqlite::{params, Connection};

use crate::types::StoredFunction;

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
