def pytest_terminal_summary(terminalreporter, exitstatus, config):
    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))
    errors = len(terminalreporter.stats.get("error", []))
    xfailed = len(terminalreporter.stats.get("xfailed", []))
    xpassed = len(terminalreporter.stats.get("xpassed", []))

    terminalreporter.write_sep("=", "Backtest / Leakage Test Summary")
    terminalreporter.write_line(f"✅ Passed : {passed}")
    terminalreporter.write_line(f"❌ Failed : {failed}")
    terminalreporter.write_line(f"💥 Errors : {errors}")
    terminalreporter.write_line(f"⏭️ Skipped: {skipped}")
    terminalreporter.write_line(f"🟡 XFail  : {xfailed}")
    terminalreporter.write_line(f"🟣 XPass  : {xpassed}")

    if failed or errors:
        terminalreporter.write_sep("-", "Action")
        terminalreporter.write_line("Zkontroluj traceback výše — testy mají custom hlášky (look-ahead / leakage / fold boundaries).")
