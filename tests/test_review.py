"""The review loop against a stub client: proposals are validated, stored append-only, and decided by a person."""
from types import SimpleNamespace as NS

import pytest

from topoflow import grid
from topoflow.orchestrator import collect
from topoflow.review import decide, list_proposals, run_review
from topoflow.sources.synthetic import make_tools


def block_tool(i, name, inp):
    return NS(type="tool_use", id=f"t{i}", name=name, input=inp)


class StubMessages:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    def create(self, **kw):
        self.seen.append(kw)
        return self.script.pop(0)


def good(target_kind, target, text="the regional mean is 1.22x the background, worth a look"):
    return {"type": "flag", "target_kind": target_kind, "target": target, "severity": "medium", "text": text,
            "suggestion": "compare against TROPOMI over the same window", "rationale": "a conversion bias would shift every anomaly"}


@pytest.fixture
def run_dir(tmp_path):
    req = grid.make_request("permian", 0.5, "2025-07-01", "2025-07-01", [0, 12])
    return collect(req, "replay", out_root=tmp_path / "runs", fixtures_dir=tmp_path / "none", cache_dir=tmp_path / "c",
                   tools=make_tools(doctored=True), progress=lambda _: None)


def test_review_stores_only_valid_proposals_and_reads_untrusted_text_as_data(run_dir):
    script = [
        NS(stop_reason="tool_use", content=[block_tool(1, "past_decisions", {}), block_tool(2, "read_source_text", {"source_id": "local_ch4"})]),
        NS(stop_reason="tool_use", content=[
            block_tool(3, "propose", good("field", "ch4.surface_ppb")),
            block_tool(4, "propose", good("field", "ch4.made_up")),         # unknown field: refused
            block_tool(5, "propose", {**good("run", "run"), "text": "short"}),  # too short: refused
        ]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="One staged file tried to give instructions; flagged.")]),
    ]
    stub = StubMessages(script)
    out = run_review(run_dir, client=NS(beta=NS(messages=stub)))
    assert "1 proposals stored" in out
    props = list_proposals(run_dir)
    assert len(props) == 1 and props[0]["target"] == "ch4.surface_ppb"
    # The model saw the injected text only as a tool result wrapped as untrusted data.
    tool_results = [r for m in stub.seen[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                    for r in m["content"]]
    assert any("untrusted_data" in r["content"] and "IMPORTANT SYSTEM NOTE" in r["content"] for r in tool_results)
    # Every request asked for the fallback and an explicit effort.
    assert all(kw["fallbacks"] == "default" and kw["output_config"]["effort"] == "high" for kw in stub.seen)


def test_decisions_are_attributed_and_final(run_dir):
    stub = StubMessages([
        NS(stop_reason="tool_use", content=[block_tool(1, "propose", good("source", "noaa_gml"))]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="done")]),
    ])
    run_review(run_dir, client=NS(beta=NS(messages=stub)))
    pid = list_proposals(run_dir)[0]["id"]
    decide(run_dir, pid, "reject", "tester", "background is fine for a one-week window")
    assert list_proposals(run_dir)[0]["decision"]["decided_by"] == "tester"
    with pytest.raises(SystemExit):
        decide(run_dir, pid, "accept", "someone else", "changed my mind")


def test_refusal_stops_the_loop(run_dir):
    stub = StubMessages([NS(stop_reason="refusal", content=[])])
    run_review(run_dir, client=NS(beta=NS(messages=stub)))
    assert list_proposals(run_dir) == []
