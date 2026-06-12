"""Neo4j schema — constraints and indexes."""

from __future__ import annotations

from blueprint_kg.graph.neo4j_client import Neo4jClient


CONSTRAINTS = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Sheet) REQUIRE (s.project_id, s.number) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Space) REQUIRE (s.project_id, s.room_number) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:PartitionType) REQUIRE (p.project_id, p.code) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:FinishCode) REQUIRE (f.project_id, f.code) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Door) REQUIRE (d.project_id, d.door_number) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Equipment) REQUIRE (e.project_id, e.code) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (s:SpecSection) REQUIRE (s.project_id, s.section_number) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Abbreviation) REQUIRE (a.project_id, a.short) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Detail) REQUIRE (d.project_id, d.id) IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (f:FixtureTag) REQUIRE (f.project_id, f.code) IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS FOR (s:Space) ON (s.project_id)",
    "CREATE INDEX IF NOT EXISTS FOR (s:Sheet) ON (s.project_id)",
    "CREATE INDEX IF NOT EXISTS FOR (f:FinishCode) ON (f.category)",
    "CREATE INDEX IF NOT EXISTS FOR (f:FinishCode) ON (f.manufacturer)",
    "CREATE INDEX IF NOT EXISTS FOR (s:Space) ON (s.name)",
]


def create_schema(client: Neo4jClient):
    for constraint in CONSTRAINTS:
        client.run_query_no_return(constraint)
    for index in INDEXES:
        client.run_query_no_return(index)