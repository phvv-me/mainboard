"""Rules over how Pydantic models validate (mcmr's PY-PYDA validation family)."""

from ....catalog import Example, Rule


class SingleFieldModelValidator(Rule, id="PY-PYDA0001", applicability="review"):
    """Model validators that depend on only one declared field."""

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT v.fact_id, count(*) AS value FROM pydantic_model_models_validators v
          JOIN (SELECT parent_id, count(DISTINCT string_value) AS distinct_field_count
                FROM pydantic_model_values WHERE relation = 'models.validators.fields_read'
                GROUP BY parent_id) r ON r.parent_id = v.record_id
          WHERE v.kind = 'model_after' AND v.fields_read_present AND v.fields_read_length > 0
            AND r.distinct_field_count = 1 AND NOT v.has_self_call AND NOT v.has_nonfield_access
          GROUP BY v.fact_id) c ON c.fact_id = f.fact_id
    """
    message = "single field model validator is {value} for `{fact_id}`"
    # `@model_validator(mode="after")` over a method reading one field becomes
    # `@field_validator("b")` over a `@classmethod` taking the field, each `self.b` and the
    # `return self` becoming the field's name and `-> Self` the field's annotation. The model's
    # validators are fixed whole or none. One `rewrite-node` row per node it rewrites.
    fix = r"""
        WITH flagged AS (
          SELECT f.finding_key, f.path, p.file_id, d.method_node AS fn, d.receiver,
                 d.record_id AS model,
                 (SELECT any_value(x.string_value) FROM pydantic_model_values x
                  WHERE x.relation = 'models.validators.fields_read'
                    AND x.parent_id = d.validator_id) AS field
          FROM finding f
          JOIN core_file p ON p.path = f.path
          JOIN pydantic_model_validator_defs d ON d.fact_id = f.fact_id
          JOIN pydantic_model_models_validators v ON v.record_id = d.validator_id
          WHERE v.kind = 'model_after' AND v.fields_read_present AND v.fields_read_length = 1
            AND NOT v.has_self_call AND NOT v.has_nonfield_access
        ), shaped AS (
          SELECT v.*, f.start_line AS def_line, f.start_column AS def_column, dc.node_id AS call,
                 if(cal.type = 'identifier', cal.node_id, name.node_id) AS callee,
                 cal.type = 'identifier' AS imported, kw.node_id AS keyword,
                 me.node_id AS holder, ty.node_id AS returned, bl.node_id AS body,
                 bl.descendant_count AS size, a.annotation, df.node_id AS def_node
          FROM flagged v
          JOIN core_node f ON f.file_id = v.file_id AND f.node_id = v.fn
          JOIN core_node df ON df.file_id = v.file_id AND df.parent_id = f.node_id
            AND df.type = 'def'
          JOIN core_node w ON w.file_id = v.file_id AND w.node_id = f.parent_id
            AND w.type = 'decorated_definition' AND w.children_count = 2
          JOIN core_node dec ON dec.file_id = v.file_id AND dec.parent_id = w.node_id
            AND dec.type = 'decorator'
          JOIN core_node dc ON dc.file_id = v.file_id AND dc.parent_id = dec.node_id
            AND dc.type = 'call'
          JOIN core_node cal ON cal.file_id = v.file_id AND cal.parent_id = dc.node_id
            AND cal.sibling_index = 0 AND cal.type IN ('identifier', 'attribute')
          LEFT JOIN core_node name ON name.file_id = v.file_id AND name.parent_id = cal.node_id
            AND name.sibling_index = 1 AND cal.type = 'attribute'
          JOIN core_node al ON al.file_id = v.file_id AND al.parent_id = dc.node_id
            AND al.type = 'argument_list' AND al.children_count = 3
          JOIN core_node kw ON kw.file_id = v.file_id AND kw.parent_id = al.node_id
            AND kw.type = 'keyword_argument'
          JOIN core_node ps ON ps.file_id = v.file_id AND ps.parent_id = f.node_id
            AND ps.type = 'parameters' AND ps.children_count = 3
          JOIN core_node me ON me.file_id = v.file_id AND me.parent_id = ps.node_id
            AND me.type = 'identifier'
          JOIN core_node bl ON bl.file_id = v.file_id AND bl.parent_id = f.node_id
            AND bl.type = 'block'
          LEFT JOIN core_node ar ON ar.file_id = v.file_id AND ar.parent_id = f.node_id
            AND ar.type = '->'
          LEFT JOIN core_node ty ON ty.file_id = v.file_id AND ty.parent_id = f.node_id
            AND ty.type = 'type' AND ty.name = 'Self'
          LEFT JOIN pydantic_model_models_fields a ON a.parent_id = v.model AND a.name = v.field
          WHERE (ar.node_id IS NULL OR (ty.node_id IS NOT NULL AND a.annotation IS NOT NULL))
        ), reads AS (
          SELECT s.finding_key, s.fn, p.node_id AS read, p.parent_id AS holder, p.sibling_index
          FROM shaped s
          JOIN core_node n ON n.file_id = s.file_id AND n.type = 'identifier' AND n.name = 'self'
            AND n.node_id BETWEEN s.body AND s.body + s.size
          JOIN core_node p ON p.file_id = s.file_id AND p.node_id = n.parent_id
            AND p.type = 'attribute' AND n.sibling_index = 0 AND p.name = 'self.' || s.field
        ), echoes AS (
          SELECT s.finding_key, s.fn, n.node_id AS value
          FROM shaped s
          JOIN core_node r ON r.file_id = s.file_id AND r.type = 'return_statement'
            AND r.node_id BETWEEN s.body AND s.body + s.size AND r.children_count = 2
          JOIN core_node n ON n.file_id = s.file_id AND n.parent_id = r.node_id
            AND n.sibling_index = 1 AND n.type = 'identifier' AND n.name = 'self'
        ), sound AS (
          SELECT s.finding_key, s.fn FROM shaped s
          WHERE (SELECT count(*) FROM echoes x WHERE x.finding_key = s.finding_key AND x.fn = s.fn) >= 1
            AND (SELECT count(*) FROM core_node r
                 WHERE r.file_id = s.file_id AND r.type = 'return_statement'
                   AND r.node_id BETWEEN s.body AND s.body + s.size)
                = (SELECT count(*) FROM echoes x WHERE x.finding_key = s.finding_key AND x.fn = s.fn)
            AND (SELECT count(*) FROM core_node n
                 WHERE n.file_id = s.file_id AND n.type = 'identifier' AND n.name = 'self'
                   AND n.node_id BETWEEN s.body AND s.body + s.size)
                = (SELECT count(*) FROM echoes x WHERE x.finding_key = s.finding_key AND x.fn = s.fn)
                  + (SELECT count(*) FROM reads x WHERE x.finding_key = s.finding_key AND x.fn = s.fn)
            AND (SELECT count(*) FROM core_node n
                 WHERE n.file_id = s.file_id AND n.type = 'identifier'
                   AND n.name IN (s.field, 'cls')
                   AND n.node_id BETWEEN s.body AND s.body + s.size)
                = (SELECT count(*) FROM reads x WHERE x.finding_key = s.finding_key AND x.fn = s.fn)
            AND NOT EXISTS (SELECT 1 FROM core_node n
                            WHERE n.file_id = s.file_id AND n.node_id BETWEEN s.body AND s.body + s.size
                              AND n.type IN ('function_definition', 'class_definition', 'lambda'))
            AND NOT EXISTS (SELECT 1 FROM reads x JOIN core_node q ON q.file_id = s.file_id
                              AND q.node_id = x.holder
                            WHERE x.finding_key = s.finding_key AND x.fn = s.fn AND (
                              q.type IN ('augmented_assignment', 'delete_statement', 'pattern_list',
                                         'tuple_pattern', 'list_pattern', 'for_in_clause')
                              OR (q.type IN ('assignment', 'for_statement') AND x.sibling_index = 0)
                              OR (q.type = 'for_statement' AND x.sibling_index = 1)))
            AND regexp_matches(substr((SELECT l.text FROM source_lines() l
                                       WHERE l.path = s.path AND l.line = s.def_line), 1,
                                      s.def_column), '^ *$')
        ), whole AS (
          SELECT s.finding_key FROM shaped s LEFT JOIN sound USING (finding_key, fn)
          GROUP BY s.finding_key HAVING bool_and(sound.fn IS NOT NULL)
             AND count(*) = (SELECT count(*) FROM flagged x WHERE x.finding_key = s.finding_key)
        ), written AS (
          SELECT s.finding_key, s.path, s.callee AS node, 'field_validator' AS template,
                 if(s.imported, ['from pydantic import field_validator'], []::VARCHAR[]) AS imports
          FROM shaped s JOIN whole USING (finding_key)
          UNION ALL
          SELECT s.finding_key, s.path, s.keyword, '"' || s.field || '"', []::VARCHAR[]
          FROM shaped s JOIN whole USING (finding_key)
          UNION ALL
          SELECT s.finding_key, s.path, s.def_node, '@classmethod' || chr(10) || 'def',
                 []::VARCHAR[]
          FROM shaped s JOIN whole USING (finding_key)
          UNION ALL
          SELECT s.finding_key, s.path, s.holder,
                 'cls, ' || s.field || if(s.returned IS NULL, '', ': ' || s.annotation),
                 []::VARCHAR[]
          FROM shaped s JOIN whole USING (finding_key)
          UNION ALL
          SELECT s.finding_key, s.path, s.returned, s.annotation, []::VARCHAR[]
          FROM shaped s JOIN whole USING (finding_key) WHERE s.returned IS NOT NULL
          UNION ALL
          SELECT s.finding_key, s.path, x.node, s.field, []::VARCHAR[]
          FROM shaped s JOIN whole USING (finding_key)
          JOIN (SELECT finding_key, fn, read AS node FROM reads
                UNION ALL
                SELECT finding_key, fn, value FROM echoes) x
            ON x.finding_key = s.finding_key AND x.fn = s.fn
        )
        SELECT finding_key, 'rewrite-node' AS operation, path, node,
               MAP {'OLD': [node]} AS captures, template, imports
        FROM written
    """
    examples = (
        Example(
            bad=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def check(self):
        if self.b > 10:
            raise ValueError("big")
        return self
"""
            ),
            good=(
                """from pydantic import BaseModel, field_validator


class Box(BaseModel):
    a: int
    b: int

    @field_validator("b")
    @classmethod
    def check(cls, b):
        if b > 10:
            raise ValueError("big")
        return b
"""
            ),
        ),
        Example(
            bad=(
                """from typing import Self

import pydantic


class Box(pydantic.BaseModel):
    a: int
    b: int = 3

    @pydantic.model_validator(mode="after")
    def check(self) -> Self:
        if self.b > 10:
            raise ValueError(f"big {self.b}")
        return self
"""
            ),
            good=(
                """import pydantic


class Box(pydantic.BaseModel):
    a: int
    b: int = 3

    @pydantic.field_validator("b")
    @classmethod
    def check(cls, b: int) -> int:
        if b > 10:
            raise ValueError(f"big {b}")
        return b
"""
            ),
        ),
        Example(
            bad=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: str
    c: int

    @model_validator(mode="after")
    def clean(self):
        self.b = self.b.strip()
        return self

    @model_validator(mode="after")
    def check(self):
        if self.c > 10:
            raise ValueError("big")
        return self
"""
            ),
            good=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: str
    c: int

    @model_validator(mode="after")
    def clean(self):
        self.b = self.b.strip()
        return self

    @model_validator(mode="after")
    def check(self):
        if self.c > 10:
            raise ValueError("big")
        return self
"""
            ),
        ),
        Example(
            bad=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def check(self):
        if self.b > 10:
            report(self)
        return self
"""
            ),
            good=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def check(self):
        if self.b > 10:
            report(self)
        return self
"""
            ),
        ),
        Example(
            bad=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def check(self):
        b = 5
        if self.b > b:
            raise ValueError("big")
        return self
"""
            ),
            good=(
                """from pydantic import BaseModel, model_validator


class Box(BaseModel):
    a: int
    b: int

    @model_validator(mode="after")
    def check(self):
        b = 5
        if self.b > b:
            raise ValueError("big")
        return self
"""
            ),
        ),
    )


class DeclarativeFieldConstraintCandidate(Rule, id="PY-PYDA0002"):
    """Field validators that reimplement built-in Pydantic constraints."""

    query = """
        SELECT f.*, c.value FROM pydantic_model_facts f JOIN (
          SELECT fact_id, sum(declarative_constraint_count) AS value
          FROM pydantic_model_models_validators WHERE kind = 'field' GROUP BY fact_id
        ) c ON c.fact_id = f.fact_id WHERE c.value > 0
    """
    message = "declarative field constraint candidate is {value} for `{fact_id}`"


class ImperativeModelInputValidation(Rule, id="PY-PYDA0003"):
    """Model factories that manually reproduce Pydantic field validation."""

    query = """
        SELECT * FROM function_functions
        WHERE is_model_method AND NOT is_pydantic_validator AND checks_raw_input_type
          AND raises_validation_exception AND constructs_owner_model
    """
    message = (
        "`{fact_id}` checks raw input types by hand and raises a validation error before it "
        "constructs its own model"
    )
