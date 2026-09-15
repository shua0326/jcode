/// The background await notification is delivered as a coordinator turn prompt,
/// so its size must not scale with swarm width. Regression guard for the
/// delivery path dropping the shared report budget.
#[test]
fn background_await_notification_stays_bounded_and_names_the_escape_hatch() {
    use crate::protocol::MAX_AWAITED_REPORT_TOTAL_CHARS;
    use crate::server::comm_await::background_completion_notification;

    let report = "R".repeat(crate::protocol::MAX_AWAITED_REPORT_SHARE_CHARS);
    let members: Vec<crate::protocol::AwaitedMemberStatus> = (0..8)
        .map(|index| crate::protocol::AwaitedMemberStatus {
            session_id: format!("session_worker_{index:02}"),
            friendly_name: Some(format!("worker-{index:02}")),
            status: "ready".to_string(),
            done: true,
            completion_report: Some(report.clone()),
        })
        .collect();

    let notification =
        background_completion_notification(true, "All 8 members are done", &members);

    // Untrimmed this notification carries 8 * 4000 report chars plus headers.
    assert!(
        notification.chars().count() <= MAX_AWAITED_REPORT_TOTAL_CHARS + 1500,
        "notification was {} chars, which reintroduces the unbounded aggregate",
        notification.chars().count()
    );
    assert_eq!(notification.matches("chars omitted").count(), 8);
    assert!(
        notification.contains("full_reports=true"),
        "a trimmed coordinator notification must say how to fetch full reports"
    );
    assert!(notification.starts_with("🐝 **Swarm await finished**"));
}

#[test]
fn background_await_notification_leaves_small_swarms_untouched() {
    use crate::server::comm_await::background_completion_notification;

    let members: Vec<crate::protocol::AwaitedMemberStatus> = (0..2)
        .map(|index| crate::protocol::AwaitedMemberStatus {
            session_id: format!("session_worker_{index:02}"),
            friendly_name: Some(format!("worker-{index:02}")),
            status: "ready".to_string(),
            done: true,
            completion_report: Some(format!("Outcome: finished slice {index}.\n\nValidation: cargo test (12 passed)")),
        })
        .collect();

    let notification =
        background_completion_notification(true, "All 2 members are done", &members);

    assert!(
        !notification.contains("chars omitted"),
        "small swarms must keep full reports verbatim: {notification}"
    );
    assert!(!notification.contains("full_reports=true"));
    assert!(notification.contains("Validation: cargo test (12 passed)"));
}
