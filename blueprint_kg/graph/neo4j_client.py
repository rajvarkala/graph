"""Neo4j client — connection management and query execution."""

from __future__ import annotations

from neo4j import GraphDatabase


class Neo4jClient:
    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "blueprint123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: dict | None = None) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_query_no_return(self, query: str, parameters: dict | None = None):
        with self.driver.session() as session:
            session.run(query, parameters or {})

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()