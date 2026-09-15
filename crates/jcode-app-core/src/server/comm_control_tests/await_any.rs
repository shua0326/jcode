#[tokio::test]
async fn await_members_any_mode_returns_after_first_match() {
    let (_env, _runtime) = RuntimeEnvGuard::new();
    let swarm_id = "swarm-any";
    let requester = "req";
    let peer_a = "peer-a";
    let peer_b = "peer-b";
    let await_runtime = AwaitMembersRuntime::default();

    let (client_tx, mut client_rx) = mpsc::unbounded_channel();
    let swarm_members = Arc::new(RwLock::new(HashMap::from([
        (requester.to_string(), member(requester, swarm_id, "ready")),
        (peer_a.to_string(), member(peer_a, swarm_id, "running")),
        (peer_b.to_string(), member(peer_b, swarm_id, "running")),
    ])));
    let swarms_by_id = Arc::new(RwLock::new(HashMap::from([(
        swarm_id.to_string(),
        HashSet::from([
            requester.to_string(),
            peer_a.to_string(),
            peer_b.to_string(),
        ]),
    )])));
    let (swarm_event_tx, _swarm_event_rx) = broadcast::channel(32);

    handle_comm_await_members(
        1,
        requester.to_string(),
        vec!["completed".to_string()],
        vec![],
        Some("any".to_string()),
        Some(60),
        false,
        false,
        false,
        CommAwaitMembersContext {
            client_event_tx: &client_tx,
            swarm_members: &swarm_members,
            swarms_by_id: &swarms_by_id,
            swarm_event_tx: &swarm_event_tx,
            await_members_runtime: &await_runtime,
        },
    )
    .await;

    {
        let mut members = swarm_members.write().await;
        members.get_mut(peer_a).expect("peer a exists").status = "completed".to_string();
    }
    let _ = swarm_event_tx.send(swarm_event(
        peer_a,
        swarm_id,
        SwarmEventType::StatusChange {
            old_status: "running".to_string(),
            new_status: "completed".to_string(),
        },
    ));

    let response = tokio::time::timeout(Duration::from_secs(1), client_rx.recv())
        .await
        .expect("response should arrive")
        .expect("channel should stay open");

    match response {
        ServerEvent::CommAwaitMembersResponse {
            completed,
            members,
            summary,
            ..
        } => {
            assert!(
                completed,
                "await any should complete after first member matches"
            );
            assert!(
                summary.contains("peer-a"),
                "summary should mention matched member"
            );
            let done_members: Vec<_> = members.into_iter().filter(|member| member.done).collect();
            assert_eq!(done_members.len(), 1);
            assert_eq!(done_members[0].session_id, peer_a);
        }
        other => panic!("expected CommAwaitMembersResponse, got {other:?}"),
    }
}

#[test]
fn await_any_is_a_dead_end_only_when_no_member_can_still_match() {
    use crate::protocol::AwaitedMemberStatus;
    use crate::server::comm_await::{mode_satisfied, wait_is_dead_end};

    let status = |state: &str, done: bool| AwaitedMemberStatus {
        session_id: format!("peer-{state}-{done}"),
        friendly_name: None,
        status: state.to_string(),
        done,
        completion_report: None,
    };
    let stopped = status("stopped", false);
    let running = status("running", false);
    let completed = status("completed", true);

    // One stopped sibling must not cancel a wait any live member can satisfy.
    assert!(!wait_is_dead_end(
        &[stopped.clone(), running.clone()],
        Some("any")
    ));
    assert!(!mode_satisfied(
        &[stopped.clone(), running.clone()],
        Some("any")
    ));

    // Every member is now unable to reach the target status.
    assert!(wait_is_dead_end(
        &[stopped.clone(), stopped.clone()],
        Some("any")
    ));

    // The default ("all") mode ends as soon as one member can never match.
    assert!(wait_is_dead_end(&[stopped.clone(), running], None));

    // A satisfied wait is never a dead end, even beside a stopped sibling.
    assert!(!wait_is_dead_end(&[stopped, completed], Some("any")));
}

#[tokio::test]
async fn await_any_background_watch_survives_a_stopped_sibling() {
    let (_env, _runtime_dir) = RuntimeEnvGuard::new();
    let swarm_id = "swarm-any-dead-end";
    let requester = "req-any-dead-end";
    let stopped = "peer-stopped";
    let live = "peer-live";
    let key = crate::server::await_members_state::request_key(
        requester,
        swarm_id,
        &[],
        &["completed".to_string()],
        Some("any"),
    );
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    crate::server::await_members_state::save_state(
        &crate::server::await_members_state::PersistedAwaitMembersState {
            key: key.clone(),
            session_id: requester.to_string(),
            swarm_id: swarm_id.to_string(),
            target_status: vec!["completed".to_string()],
            requested_ids: vec![],
            mode: Some("any".to_string()),
            created_at_unix_ms: now_ms,
            deadline_unix_ms: now_ms.saturating_add(120_000),
            background: true,
            notify: true,
            wake: true,
            final_response: None,
        },
    );

    let await_runtime = AwaitMembersRuntime::default();
    let swarm_members = Arc::new(RwLock::new(HashMap::from([
        (requester.to_string(), member(requester, swarm_id, "ready")),
        (stopped.to_string(), member(stopped, swarm_id, "stopped")),
        (live.to_string(), member(live, swarm_id, "running")),
    ])));
    let swarms_by_id = Arc::new(RwLock::new(HashMap::from([(
        swarm_id.to_string(),
        HashSet::from([
            requester.to_string(),
            stopped.to_string(),
            live.to_string(),
        ]),
    )])));
    let (swarm_event_tx, _swarm_event_rx) = broadcast::channel(32);

    let mut bus_rx = crate::bus::Bus::global().subscribe();

    crate::server::comm_await::resume_background_awaits(
        &swarm_members,
        &swarms_by_id,
        &swarm_event_tx,
        &await_runtime,
    )
    .await;

    tokio::time::sleep(Duration::from_millis(150)).await;
    let pending = crate::server::await_members_state::load_state(&key)
        .expect("a satisfiable any-mode watch must stay persisted");
    assert!(
        pending.final_response.is_none(),
        "a stopped sibling must not finalize a wait another member can satisfy"
    );
    while let Ok(event) = bus_rx.try_recv() {
        assert!(
            !matches!(event, crate::bus::BusEvent::SwarmAwaitCompleted(ref event) if event.session_id == requester),
            "no completion should be published while a live member remains"
        );
    }

    {
        let mut members = swarm_members.write().await;
        members.get_mut(live).expect("live member exists").status = "completed".to_string();
    }
    let _ = swarm_event_tx.send(swarm_event(
        live,
        swarm_id,
        SwarmEventType::StatusChange {
            old_status: "running".to_string(),
            new_status: "completed".to_string(),
        },
    ));

    let event = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match bus_rx.recv().await {
                Ok(crate::bus::BusEvent::SwarmAwaitCompleted(event))
                    if event.session_id == requester =>
                {
                    return event;
                }
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => {
                    panic!("bus closed before SwarmAwaitCompleted arrived")
                }
            }
        }
    })
    .await
    .expect("any-mode watch should still wake when a live member completes");

    assert!(
        event.completed,
        "the watch should report completion once a member matches"
    );
    assert!(event.notify);
    assert!(event.wake);
    let final_state = crate::server::await_members_state::load_state(&key)
        .expect("state should still be persisted");
    assert!(
        final_state
            .final_response
            .expect("watch should persist a final response")
            .completed
    );
}
