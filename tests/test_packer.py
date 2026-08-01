from llm_ctxpack import ContextItem, count_tokens, pack


def test_all_items_fit_when_budget_is_large():
    items = [
        ContextItem(id="a", text="hello world " * 5, priority=1),
        ContextItem(id="b", text="goodbye world " * 5, priority=1),
    ]
    result = pack(items, budget=10_000)
    assert len(result.items) == 2
    assert result.dropped_ids == []
    assert result.total_tokens <= result.budget


def test_higher_priority_wins_when_budget_is_tight():
    important = ContextItem(id="important", text="critical info. " * 200, priority=10, truncatable=False)
    filler = ContextItem(id="filler", text="filler text. " * 200, priority=1, truncatable=False)
    budget = count_tokens(important.text) + 5  # not enough room for both
    result = pack([filler, important], budget=budget)
    ids = [i.id for i in result.items]
    assert "important" in ids
    assert "filler" in result.dropped_ids


def test_truncation_respects_min_tokens():
    item = ContextItem(id="doc", text="word " * 1000, priority=1, truncatable=True, min_tokens=500)
    # Budget can't satisfy min_tokens, so it should be dropped, not truncated to mush.
    result = pack([item], budget=10)
    assert result.items == []
    assert result.dropped_ids == ["doc"]


def test_truncation_happens_when_budget_allows_min_tokens():
    item = ContextItem(id="doc", text="word " * 1000, priority=1, truncatable=True, min_tokens=10)
    result = pack([item], budget=50)
    assert len(result.items) == 1
    assert result.items[0].truncated is True
    assert result.items[0].tokens <= 50


def test_render_joins_items_in_priority_order():
    items = [
        ContextItem(id="low", text="LOW", priority=1),
        ContextItem(id="high", text="HIGH", priority=5),
    ]
    result = pack(items, budget=10_000)
    rendered = result.render(separator="|")
    assert rendered.index("HIGH") < rendered.index("LOW")


def test_reserve_tokens_shrinks_effective_budget():
    item = ContextItem(id="doc", text="word " * 100, priority=1, min_tokens=1)
    result = pack([item], budget=100, reserve_tokens=90)
    assert result.budget == 10
    assert result.total_tokens <= 10


def test_count_tokens_nonempty_and_zero():
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0


def test_empty_input_produces_empty_result():
    result = pack([], budget=1000)
    assert result.items == []
    assert result.total_tokens == 0
