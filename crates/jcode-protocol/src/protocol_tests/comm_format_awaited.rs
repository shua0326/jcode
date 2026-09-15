fn awaited_member(session_id: &str, done: bool) -> AwaitedMemberStatus {
    AwaitedMemberStatus {
        session_id: session_id.to_string(),
        friendly_name: Some(session_id.to_string()),
        status: if done { "completed" } else { "running" }.to_string(),
        done,
        completion_report: None,
    }
}

fn member_with_report(session_id: &str, report: &str) -> AwaitedMemberStatus {
    AwaitedMemberStatus {
        session_id: session_id.to_string(),
        friendly_name: Some(session_id.to_string()),
        status: "completed".to_string(),
        done: true,
        completion_report: Some(report.to_string()),
    }
}

#[test]
fn awaited_members_header_all_done() {
    let members = vec![awaited_member("fox", true), awaited_member("wolf", true)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "All 2 members are done: fox, wolf",
        &members,
        &std::collections::HashMap::new(),
        None,
    );
    assert!(
        output.starts_with("All members done."),
        "expected all-done header, got: {output}"
    );
}

#[test]
fn awaited_members_header_any_mode_partial_match() {
    let members = vec![awaited_member("fox", true), awaited_member("wolf", false)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "Matched 1 member: fox",
        &members,
        &std::collections::HashMap::new(),
        None,
    );
    assert!(
        output.starts_with("Await satisfied."),
        "any-mode partial match must not claim all members are done, got: {output}"
    );
    assert!(!output.starts_with("All members done."));
}

#[test]
fn awaited_members_header_incomplete() {
    let members = vec![awaited_member("fox", false)];
    let output = format_comm_awaited_members_with_reports(
        false,
        "Timed out. Still waiting on: fox (running)",
        &members,
        &std::collections::HashMap::new(),
        None,
    );
    assert!(
        output.starts_with("Await incomplete."),
        "expected incomplete header, got: {output}"
    );
}

#[test]
fn awaited_members_without_budget_keeps_full_reports() {
    let report = "x".repeat(MAX_AWAITED_REPORT_SHARE_CHARS);
    let members = vec![member_with_report("fox", &report)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "done",
        &members,
        &std::collections::HashMap::new(),
        None,
    );
    assert!(output.contains(&report), "unbudgeted rendering must be lossless");
    assert!(!output.contains("chars omitted"));
}

#[test]
fn awaited_members_budget_keeps_a_single_report_intact() {
    let report = "y".repeat(1200);
    let members = vec![member_with_report("fox", &report)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "done",
        &members,
        &std::collections::HashMap::new(),
        Some(MAX_AWAITED_REPORT_TOTAL_CHARS),
    );
    assert!(
        output.contains(&report),
        "a report under its fair share must not be trimmed"
    );
    assert!(!output.contains("chars omitted"));
    assert!(!output.contains("full_reports=true"));
}

#[test]
fn awaited_members_budget_bounds_a_wide_swarm() {
    let report = "z".repeat(MAX_AWAITED_REPORT_SHARE_CHARS);
    let members: Vec<AwaitedMemberStatus> = (0..8)
        .map(|index| member_with_report(&format!("member-{index:02}"), &report))
        .collect();
    let output = format_comm_awaited_members_with_reports(
        true,
        "All 8 members are done",
        &members,
        &std::collections::HashMap::new(),
        Some(MAX_AWAITED_REPORT_TOTAL_CHARS),
    );

    // Untrimmed, this rendering would carry 8 * 4000 report chars. The budget
    // must keep it near the total, not merely below the per-report cap.
    let report_chars = output
        .lines()
        .filter(|line| !line.starts_with("--- ") && line.chars().all(|c| c == 'z'))
        .map(|line| line.chars().count())
        .sum::<usize>();
    assert!(
        report_chars <= MAX_AWAITED_REPORT_TOTAL_CHARS + 512,
        "budgeted report content was {report_chars} chars, over the {MAX_AWAITED_REPORT_TOTAL_CHARS} budget"
    );
    assert_eq!(
        output.matches("chars omitted").count(),
        8,
        "every trimmed member must be marked as elided"
    );
    assert!(
        output.contains("full_reports=true"),
        "a trimmed result must say how to fetch full text"
    );
    for index in 0..8 {
        assert!(
            output.contains(&format!("\"member-{index:02}\"")),
            "the escape hatch must name every trimmed member"
        );
    }
}

#[test]
fn awaited_members_elision_keeps_report_tail() {
    let head = "H".repeat(MAX_AWAITED_REPORT_SHARE_CHARS);
    let tail = "\n\nValidation:\nall tests pass\n\nFollow-ups/blockers:\nnone";
    let report = format!("{head}{tail}");
    let members = vec![member_with_report("fox", &report)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "done",
        &members,
        &std::collections::HashMap::new(),
        Some(1200),
    );
    assert!(
        output.contains("Validation:"),
        "a trimmed report must keep its trailing validation section: {output}"
    );
    assert!(output.contains("Follow-ups/blockers:"));
    assert!(output.contains("chars omitted"));
}

#[test]
fn awaited_members_elision_handles_multibyte_reports() {
    let report = "ủ".repeat(MAX_AWAITED_REPORT_SHARE_CHARS);
    let members = vec![member_with_report("fox", &report)];
    let output = format_comm_awaited_members_with_reports(
        true,
        "done",
        &members,
        &std::collections::HashMap::new(),
        Some(600),
    );
    assert!(output.contains("chars omitted"));
}

