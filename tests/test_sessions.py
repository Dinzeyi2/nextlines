from execution import resume_session, start_session, terminate_session


def test_variable_reuse_across_turns():
    session = start_session()
    ctx = session.context
    ctx.add_variable('a', 1)
    sid = session.id

    resumed = resume_session(sid)
    assert resumed is session
    resumed.context.add_variable('b', resumed.context.get_variable('a') + 1)
    assert resumed.context.get_variable('b') == 2
    terminate_session(sid)


def test_stepwise_refinement_with_undo():
    session = start_session()
    ctx = session.context
    ctx.add_variable('x', 1)
    ctx.save_state()
    ctx.add_variable('x', 2)
    assert ctx.get_variable('x') == 2
    ctx.undo()
    assert ctx.get_variable('x') == 1
    terminate_session(session.id)
