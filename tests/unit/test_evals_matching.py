from evals.matching import analyse_case, overlap, same_person


def _expected(tasks, traps=()):
    return {"tasks": tasks, "not_task_quotes": list(traps)}


def _gold(quote, assignee="Luis", due=None):
    return {"evidence_quote": quote, "assignee": assignee, "due_date": due}


def _pred(quotes, assignee="Luis", due=None):
    return {"evidence_quotes": quotes, "assignee": assignee, "due_date": due}


def test_overlap_is_measured_from_the_expected_quote():
    assert (
        overlap(
            "I'll review it right after this call",
            "Perfect, I'll review it right after this call, Sofía.",
        )
        == 1.0
    )
    assert overlap("I'll review it right after this call", "Great, thanks") < 0.2
    assert overlap("", "anything") == 0.0


def test_same_person_compares_first_names_and_empties():
    assert same_person("Luis", "luis alain")
    assert same_person(None, None)
    assert not same_person("Luis", None)
    assert not same_person("Luis", "Ana")


def test_perfect_match():
    analysis = analyse_case(
        _expected([_gold("yo me encargo del despliegue", due="2026-10-23")]),
        [_pred(["Luis: yo me encargo del despliegue"], due="2026-10-23")],
    )
    assert analysis.matches == [(0, 0)] and not analysis.missed and not analysis.extra
    assert analysis.wrong == set()


def test_missed_extra_and_wrong_fields_are_told_apart():
    expected = _expected(
        [
            _gold("cerrar la integración de autenticación", "Luis", "2026-10-23"),
            _gold("revisar el pull request 142", "Ana"),
        ]
    )
    predicted = [
        _pred(["cerrar la integración de autenticación"], assignee="Marta", due="2026-10-24"),
        _pred(["comprar café para la oficina"]),
    ]
    analysis = analyse_case(expected, predicted)
    assert analysis.matches == [(0, 0)]
    assert analysis.missed == [1]  # the pull request task was never found
    assert analysis.extra == [1]  # the coffee task was invented
    assert analysis.assignee_correct == {0: False} and analysis.date_correct == {0: False}
    assert analysis.wrong == {0, 1}


def test_each_task_is_used_once():
    expected = _expected(
        [
            _gold("revisar el pull request 142 hoy", "Carlos"),
            _gold("revisar el pull request 142 hoy", "Luis"),
        ]
    )
    predicted = [_pred(["revisar el pull request 142 hoy"], assignee="Carlos")]
    analysis = analyse_case(expected, predicted)
    assert len(analysis.matches) == 1 and len(analysis.missed) == 1


def test_trap_is_hit_only_by_invented_tasks():
    expected = _expected([], traps=["we should organize a team lunch before the month ends"])
    invented = analyse_case(
        expected, [_pred(["Sofía: We should organize a team lunch before the month ends"])]
    )
    assert invented.trap_hits == [(0, 0)]
    clean = analyse_case(expected, [])
    assert clean.trap_hits == []
