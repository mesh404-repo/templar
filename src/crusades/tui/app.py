"""Crusades TUI - Terminal dashboard for miners."""

import argparse
import sys
import time

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from crusades import COMPETITION_VERSION
from crusades.tui.client import (
    CrusadesClient,
    CrusadesData,
    DatabaseClient,
    MockClient,
    SubmissionDetail,
    format_time_ago,
)

console = Console(force_terminal=True)


def create_chart(history: list[dict], width: int = 50, height: int = 8) -> Text:
    """Create an ASCII LINE chart of MFU over time."""
    if not history:
        return Text("No data available", style="dim italic", justify="center")

    # Sort by timestamp and get MFU values (fallback to tps for backwards compat)
    sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))
    mfu_values = [h.get("mfu", h.get("tps", 0)) for h in sorted_history]

    if not mfu_values:
        return Text("No data available", style="dim italic", justify="center")

    min_mfu = min(mfu_values)
    max_mfu = max(mfu_values)

    # Add padding to min/max for better visualization
    if max_mfu == min_mfu:
        # All values same - show horizontal line in middle
        padding = max_mfu * 0.1 if max_mfu > 0 else 100
        min_mfu = max_mfu - padding
        max_mfu = max_mfu + padding

    mfu_range = max_mfu - min_mfu

    # Chart dimensions
    chart_width = width - 7  # Account for y-axis label

    # Create a 2D grid for the chart
    grid = [[" " for _ in range(chart_width)] for _ in range(height)]

    # Calculate positions for each data point
    if len(mfu_values) == 1:
        # Single point - show in middle
        x_positions = [chart_width // 2]
    else:
        x_positions = [
            int(i * (chart_width - 1) / (len(mfu_values) - 1)) for i in range(len(mfu_values))
        ]

    y_positions = []
    for mfu in mfu_values:
        normalized = (mfu - min_mfu) / mfu_range
        y = int(normalized * (height - 1))
        y = max(0, min(height - 1, y))
        y_positions.append(y)

    # Draw connecting lines between points
    for i in range(len(mfu_values) - 1):
        x1, y1 = x_positions[i], y_positions[i]
        x2, y2 = x_positions[i + 1], y_positions[i + 1]

        # Draw line between points using Bresenham-like approach
        dx = x2 - x1
        dy = y2 - y1
        steps = max(abs(dx), abs(dy), 1)

        for step in range(steps + 1):
            t = step / steps if steps > 0 else 0
            x = int(x1 + t * dx)
            y = int(y1 + t * dy)
            if 0 <= x < chart_width and 0 <= y < height:
                # Use different characters based on slope
                if dy > 0 and step > 0 and step < steps:
                    grid[height - 1 - y][x] = "╱"
                elif dy < 0 and step > 0 and step < steps:
                    grid[height - 1 - y][x] = "╲"
                else:
                    grid[height - 1 - y][x] = "─"

    # Draw data points (overwrite lines at point locations)
    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        if 0 <= x < chart_width and 0 <= y < height:
            grid[height - 1 - y][x] = "●"

    # Build output lines with y-axis labels (MFU shown as percentage)
    lines = []
    for row in range(height):
        if row == 0:
            label = f"{max_mfu:>4.1f}% │"
        elif row == height - 1:
            label = f"{min_mfu:>4.1f}% │"
        elif row == height // 2:
            mid_val = min_mfu + mfu_range / 2
            label = f"{mid_val:>4.1f}% │"
        else:
            label = "       │"
        lines.append(label + "".join(grid[row]))

    # X-axis
    x_axis = "      └" + "─" * chart_width
    lines.append(x_axis)

    # Time labels
    if sorted_history:
        first_time = sorted_history[0].get("timestamp", "")[:10]
        last_time = sorted_history[-1].get("timestamp", "")[:10]
        time_label = (
            f"       {first_time}"
            + " " * (chart_width - len(first_time) - len(last_time))
            + last_time
        )
        lines.append(time_label)

    chart_text = Text()
    for i, line in enumerate(lines):
        if i < len(lines) - 2:  # Chart area
            chart_text.append(line[:7])  # Y-axis label
            # Color the line chart
            chart_line = line[7:]
            for char in chart_line:
                if char in "●":
                    chart_text.append(char, style="bold cyan")
                elif char in "─╱╲":
                    chart_text.append(char, style="green")
                else:
                    chart_text.append(char)
        else:
            chart_text.append(line, style="dim")
        if i < len(lines) - 1:
            chart_text.append("\n")

    return chart_text


def create_chart_panel(data: CrusadesData) -> Panel:
    """Create the MFU history chart panel."""
    # Use full width - get console width dynamically
    chart = create_chart(data.history, width=90, height=6)
    return Panel(
        chart,
        title="[bold]MFU History[/bold]",
        border_style="yellow",
    )


def create_stats_panel(data: CrusadesData) -> Panel:
    """Create the stats overview panel."""
    overview = data.overview
    threshold = data.threshold or {}

    stats = Table.grid(padding=(0, 4))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")

    # Calculate MFU to Beat: top_mfu * (1 + threshold)
    # Handle None values to avoid math errors
    top_mfu = overview.get("current_top_score") or 0
    threshold_val = threshold.get("decayed_threshold") or 0.01
    mfu_to_beat = top_mfu * (1 + threshold_val) if top_mfu > 0 else 0

    stats.add_row(
        Text(f"24h Submissions\n{overview.get('submissions_24h', 0)}", justify="center"),
        Text(f"Top MFU\n{top_mfu:.1f}%", justify="center"),
        Text(f"MFU to Beat\n{mfu_to_beat:.1f}%", justify="center"),
        Text(f"Active Miners\n{overview.get('active_miners', 0)}", justify="center"),
        Text(f"Total\n{overview.get('total_submissions', 0)}", justify="center"),
    )

    return Panel(Align.center(stats), title="[bold]Stats[/bold]", border_style="blue")


def create_validator_panel(data: CrusadesData) -> Panel:
    """Create the validator status panel."""
    validator = data.validator
    queue = data.queue

    status = validator.get("status", "unknown")
    if status == "running":
        status_color = "green"
    elif status == "idle":
        status_color = "yellow"
    else:
        status_color = "red"
    status_dot = f"[{status_color}]●[/{status_color}]"

    current_eval = validator.get("current_evaluation")
    eval_text = "None"
    if current_eval:
        # Handle both dict (legacy) and string (new) formats
        if isinstance(current_eval, dict) and current_eval.get("miner_uid") is not None:
            eval_text = f"UID {current_eval['miner_uid']}"
        elif isinstance(current_eval, str):
            eval_text = current_eval[:20] + "..." if len(current_eval) > 20 else current_eval

    uptime = validator.get("uptime", "N/A")
    evals_1h = validator.get("evaluations_completed_1h", 0)

    # Use queued_count if available, fallback to pending_count for backwards compat
    queued = queue.get("queued_count", queue.get("pending_count", 0))
    running = queue.get("running_count", 0)
    finished = queue.get("finished_count", 0)
    failed = queue.get("failed_count", 0)

    info = (
        f"{status_dot} {status.upper()}  |  "
        f"Evaluating: {eval_text}  |  "
        f"Evals/hr: {evals_1h}  |  "
        f"[yellow]Queued: {queued}[/]  |  "
        f"[cyan]Running: {running}[/]  |  "
        f"[green]Done: {finished}[/]  |  "
        f"[red]Failed: {failed}[/]  |  "
        f"Success: {uptime}"
    )

    return Panel(
        Text.from_markup(info, justify="center"),
        title="[bold]Validator[/bold]",
        border_style="cyan",
    )


def create_leaderboard_table(
    data: CrusadesData, selected_idx: int | None = None, active: bool = True
) -> Panel:
    """Create the leaderboard table with selection highlight."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold")

    table.add_column("#", justify="right", width=3)
    table.add_column("Rank", justify="right", width=5)
    table.add_column("UID", justify="right", width=6)
    table.add_column("MFU", justify="right", style="green", width=10)
    table.add_column("Evals", justify="right", width=6)
    table.add_column("Submitted", justify="right", width=10)

    for idx, entry in enumerate(data.leaderboard[:10]):
        rank = entry.get("rank", "?")
        rank_style = ""
        if rank == 1:
            rank_style = "[yellow]"
        elif rank == 2:
            rank_style = "[white]"
        elif rank == 3:
            rank_style = "[#cd7f32]"

        rank_display = f"{rank_style}{rank}[/]" if rank_style else str(rank)

        # Highlight selected row
        is_selected = active and selected_idx == idx
        row_style = "reverse" if is_selected else ""

        table.add_row(
            f"[cyan]{idx + 1}[/]",
            rank_display,
            str(entry.get("miner_uid", "?")),
            f"{entry.get('final_score', 0):.2f}",
            str(entry.get("num_evaluations", 0)),
            format_time_ago(entry.get("created_at")),
            style=row_style,
        )

    if not data.leaderboard:
        table.add_row("-", "-", "-", "-", "-", "-")

    border = "green" if active else "dim"
    return Panel(table, title="[bold]Leaderboard[/bold]", border_style=border)


def create_recent_table(
    data: CrusadesData, selected_idx: int | None = None, active: bool = False
) -> Panel:
    """Create the recent submissions table with selection highlight."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold")

    table.add_column("#", justify="right", width=3)
    table.add_column("UID", justify="right", width=6)
    table.add_column("Status", justify="left", width=16)
    table.add_column("MFU", justify="right", width=10)
    table.add_column("Submitted", justify="right", width=10)

    status_colors = {
        "pending": "yellow",
        "validating": "yellow",
        "evaluating": "cyan",
        "finished": "green",
        "failed_validation": "red",
        "failed_evaluation": "red",
        "error": "red",
    }

    for idx, entry in enumerate(data.recent[:10]):
        status = entry.get("status", "unknown")
        color = status_colors.get(status, "white")

        score = entry.get("final_score")
        score_display = f"{score:.2f}" if score and score > 0 else "-"

        is_selected = active and selected_idx == idx
        row_style = "reverse" if is_selected else ""

        table.add_row(
            f"[cyan]{idx + 1}[/]",
            str(entry.get("miner_uid", "?")),
            f"[{color}]{status}[/{color}]",
            score_display,
            format_time_ago(entry.get("created_at")),
            style=row_style,
        )

    if not data.recent:
        table.add_row("-", "-", "-", "-", "-")

    border = "magenta" if active else "dim"
    return Panel(table, title="[bold]Recent Activity[/bold]", border_style=border)


def create_dashboard_layout(
    data: CrusadesData,
    leaderboard_idx: int | None = None,
    recent_idx: int | None = None,
    active_panel: str = "leaderboard",
    demo: bool = False,
) -> Layout:
    """Create the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="stats", size=5),
        Layout(name="validator", size=3),
        Layout(name="chart", size=10),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )

    title = "τemplar crusades"
    if demo:
        title += "  [yellow][DEMO][/yellow]"

    layout["header"].update(
        Panel(
            Text.from_markup(title, justify="center", style="bold white"),
            border_style="white",
        )
    )

    layout["stats"].update(create_stats_panel(data))
    layout["validator"].update(create_validator_panel(data))
    layout["chart"].update(create_chart_panel(data))

    layout["main"].split_row(
        Layout(name="leaderboard", ratio=3),
        Layout(name="recent", ratio=2),
    )

    layout["leaderboard"].update(
        create_leaderboard_table(
            data,
            selected_idx=leaderboard_idx,
            active=(active_panel == "leaderboard"),
        )
    )
    layout["recent"].update(
        create_recent_table(
            data,
            selected_idx=recent_idx,
            active=(active_panel == "recent"),
        )
    )

    footer_text = (
        "[cyan]j/k[/] Navigate  "
        "[cyan]h/l[/] Switch panel  "
        "[cyan]Enter[/] View details  "
        "[cyan]r[/] Refresh  "
        "[cyan]q[/] Quit"
    )
    layout["footer"].update(
        Panel(Text.from_markup(footer_text, justify="center"), border_style="dim")
    )

    return layout


def create_submission_header(detail: SubmissionDetail) -> Panel:
    """Create the submission header panel."""
    sub = detail.submission

    status = sub.get("status", "unknown")
    status_colors = {
        "pending": "yellow",
        "validating": "yellow",
        "evaluating": "cyan",
        "finished": "green",
        "failed_validation": "red",
        "failed_evaluation": "red",
        "error": "red",
    }
    status_color = status_colors.get(status, "white")

    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    hotkey = sub.get("miner_hotkey", "N/A")
    if len(hotkey) > 16:
        hotkey = hotkey[:8] + "..." + hotkey[-8:]

    code_hash = sub.get("code_hash", "N/A")
    if len(code_hash) > 16:
        code_hash = code_hash[:16] + "..."

    score = sub.get("final_score")
    score_display = f"{score:.2f}" if score is not None else "N/A"

    grid.add_row(
        f"[bold]UID:[/] {sub.get('miner_uid', 'N/A')}",
        f"[bold]Status:[/] [{status_color}]{status}[/{status_color}]",
        f"[bold]MFU:[/] [green]{score_display}%[/green]",
        f"[bold]Submitted:[/] {format_time_ago(sub.get('created_at'))}",
    )
    grid.add_row(
        f"[bold]Hotkey:[/] {hotkey}",
        f"[bold]Hash:[/] {code_hash}",
        "",
        "",
    )

    error = sub.get("error_message")
    if error:
        grid.add_row(f"[red][bold]Error:[/] {error}[/red]", "", "", "")

    return Panel(grid, title="[bold]Submission Details[/bold]", border_style="blue")


def create_evaluations_table(detail: SubmissionDetail) -> Panel:
    """Create the evaluations table."""
    table = Table(expand=True, box=None)
    table.add_column("#", justify="right", width=4)
    table.add_column("MFU", justify="right", style="green", width=10)
    table.add_column("TPS", justify="right", width=10)
    table.add_column("Tokens", justify="right", width=10)
    table.add_column("Wall Time", justify="right", width=10)
    table.add_column("Status", justify="center", width=6)
    table.add_column("Time", justify="right", width=10)

    total_mfu = 0.0
    count = 0

    for idx, eval_data in enumerate(detail.evaluations, 1):
        mfu = eval_data.get("mfu", 0)
        tps = eval_data.get("tokens_per_second", 0)
        total_mfu += mfu
        count += 1

        success = eval_data.get("success", False)
        status_display = "[green]✓[/green]" if success else "[red]✗[/red]"

        table.add_row(
            str(idx),
            f"{mfu:.2f}%" if mfu else "-",
            f"{tps:.0f}",
            str(eval_data.get("total_tokens", 0)),
            f"{eval_data.get('wall_time_seconds', 0):.2f}s",
            status_display,
            format_time_ago(eval_data.get("created_at")),
        )

    if not detail.evaluations:
        table.add_row("-", "-", "-", "-", "-", "-", "-")

    avg_mfu = total_mfu / count if count > 0 else 0
    title = f"[bold]Evaluations[/bold] (Avg MFU: [green]{avg_mfu:.2f}%[/green])"

    return Panel(table, title=title, border_style="cyan")


def create_code_panel(
    detail: SubmissionDetail, scroll_offset: int = 0, visible_lines: int = 20
) -> Panel:
    """Create the syntax-highlighted code panel with scroll support."""
    code = detail.code

    if not code:
        content = Text("Code not available", style="dim italic", justify="center")
        total_lines = 0
    else:
        lines = code.split("\n")
        total_lines = len(lines)

        # Slice code based on scroll offset
        start_line = scroll_offset
        end_line = min(scroll_offset + visible_lines, total_lines)
        visible_code = "\n".join(lines[start_line:end_line])

        content = Syntax(
            visible_code,
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
            start_line=start_line + 1,  # Line numbers start from scroll position
        )

    code_hash = detail.submission.get("code_hash", "")[:16]
    title = "[bold]Code[/bold]"
    if code_hash:
        title += f" [dim](hash: {code_hash}...)[/dim]"

    # Show scroll position
    if code and total_lines > visible_lines:
        title += f" [cyan](lines {scroll_offset + 1}-{min(scroll_offset + visible_lines, total_lines)}/{total_lines}, j/k to scroll)[/cyan]"

    return Panel(content, title=title, border_style="green")


def create_submission_layout(
    detail: SubmissionDetail, show_code: bool = True, code_scroll: int = 0
) -> Layout:
    """Create the submission detail layout."""
    layout = Layout()

    if show_code:
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="details", size=5),
            Layout(name="evaluations", size=10),
            Layout(name="code"),
            Layout(name="footer", size=3),
        )
    else:
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="details", size=5),
            Layout(name="evaluations"),
            Layout(name="footer", size=3),
        )

    layout["header"].update(
        Panel(
            Text("SUBMISSION DETAILS", justify="center", style="bold white"),
            border_style="white",
        )
    )

    layout["details"].update(create_submission_header(detail))
    layout["evaluations"].update(create_evaluations_table(detail))

    if show_code:
        layout["code"].update(create_code_panel(detail, scroll_offset=code_scroll))

    footer_text = "[cyan]j/k[/] Scroll code  [cyan]c[/] Toggle code  [cyan]r[/] Refresh  [cyan]b/Esc[/] Back  [cyan]q[/] Quit"
    layout["footer"].update(
        Panel(Text.from_markup(footer_text, justify="center"), border_style="dim")
    )

    return layout


def get_key_nonblocking(timeout: float = 0.1) -> str | None:
    """Get a single keypress with timeout (Unix only)."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if sys.stdin in select.select([sys.stdin], [], [], timeout)[0]:
            ch = sys.stdin.read(1)
            # Handle escape sequences (arrow keys)
            if ch == "\x1b":
                if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                    ch += sys.stdin.read(1)
                    if ch == "\x1b[" and sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                        ch += sys.stdin.read(1)
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_tui(base_url: str, refresh_interval: int, demo: bool = False, db_path: str | None = None):
    """Run the interactive TUI."""
    import termios

    # Save terminal state
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Select client based on mode (filter by competition version)
    competition_version = COMPETITION_VERSION

    if demo:
        client_class = MockClient
        client_kwargs = {}
    elif db_path:
        client_class = DatabaseClient
        client_kwargs = {"db_path": db_path, "spec_version": competition_version}
    else:
        client_class = CrusadesClient
        client_kwargs = {"base_url": base_url}

    # Hide cursor and set up alternate screen buffer
    print("\033[?25l", end="", flush=True)  # Hide cursor
    print("\033[?1049h", end="", flush=True)  # Use alternate screen buffer

    try:
        with client_class(**client_kwargs) as client:
            # State
            data = client.fetch_all()
            current_view = "dashboard"
            current_detail: SubmissionDetail | None = None
            show_code = True
            code_scroll = 0  # Scroll offset for code view
            last_refresh = time.time()

            # Dashboard navigation state
            active_panel = "leaderboard"  # or "recent"
            leaderboard_idx = 0 if data.leaderboard else None
            recent_idx = 0 if data.recent else None
            needs_redraw = True  # Only redraw when something changes

            while True:
                # Only redraw when needed (reduces blinking)
                if needs_redraw:
                    print("\033[2J\033[H", end="", flush=True)

                    if current_view == "dashboard":
                        layout = create_dashboard_layout(
                            data,
                            leaderboard_idx=leaderboard_idx
                            if active_panel == "leaderboard"
                            else None,
                            recent_idx=recent_idx if active_panel == "recent" else None,
                            active_panel=active_panel,
                            demo=demo,
                        )
                        console.print(layout)
                    elif current_view == "submission" and current_detail:
                        layout = create_submission_layout(
                            current_detail, show_code=show_code, code_scroll=code_scroll
                        )
                        console.print(layout)

                    needs_redraw = False

                # Handle input
                key = get_key_nonblocking(0.3)

                if key:
                    needs_redraw = True  # Any keypress triggers redraw

                    # Quit
                    if key == "q" or key == "\x03":
                        break

                    elif current_view == "dashboard":
                        # Refresh
                        if key == "r":
                            data = client.fetch_all()
                            if data.leaderboard and leaderboard_idx is None:
                                leaderboard_idx = 0
                            if data.recent and recent_idx is None:
                                recent_idx = 0
                            last_refresh = time.time()

                        # Switch panels with left/right arrows, h/l, or Tab
                        elif key in ("\x1b[D", "\x1b[C", "\t", "h", "l"):  # Left, Right, Tab, h, l
                            if active_panel == "leaderboard":
                                active_panel = "recent"
                            else:
                                active_panel = "leaderboard"

                        # Navigate up (arrow or k)
                        elif key in ("\x1b[A", "k"):  # Up arrow or k
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                leaderboard_idx = max(0, leaderboard_idx - 1)
                            elif active_panel == "recent" and recent_idx is not None:
                                recent_idx = max(0, recent_idx - 1)

                        # Navigate down (arrow or j)
                        elif key in ("\x1b[B", "j"):  # Down arrow or j
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                max_idx = min(9, len(data.leaderboard) - 1)
                                leaderboard_idx = min(max_idx, leaderboard_idx + 1)
                            elif active_panel == "recent" and recent_idx is not None:
                                max_idx = min(9, len(data.recent) - 1)
                                recent_idx = min(max_idx, recent_idx + 1)

                        # Select with Enter
                        elif key in ("\r", "\n"):
                            submission_id = None
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                if leaderboard_idx < len(data.leaderboard):
                                    submission_id = data.leaderboard[leaderboard_idx].get(
                                        "submission_id"
                                    )
                            elif active_panel == "recent" and recent_idx is not None:
                                if recent_idx < len(data.recent):
                                    submission_id = data.recent[recent_idx].get("submission_id")

                            if submission_id:
                                current_detail = client.fetch_submission_detail(submission_id)
                                current_view = "submission"
                                show_code = True
                                code_scroll = 0

                        # Quick select with number keys
                        elif key.isdigit() and key != "0":
                            idx = int(key) - 1
                            if active_panel == "leaderboard":
                                if idx < len(data.leaderboard):
                                    leaderboard_idx = idx
                                    submission_id = data.leaderboard[idx].get("submission_id")
                                    if submission_id:
                                        current_detail = client.fetch_submission_detail(
                                            submission_id
                                        )
                                        current_view = "submission"
                                        show_code = True
                                        code_scroll = 0
                            else:
                                if idx < len(data.recent):
                                    recent_idx = idx
                                    submission_id = data.recent[idx].get("submission_id")
                                    if submission_id:
                                        current_detail = client.fetch_submission_detail(
                                            submission_id
                                        )
                                        current_view = "submission"
                                        show_code = True
                                        code_scroll = 0

                    elif current_view == "submission":
                        # Back to dashboard
                        if key in ("b", "\x1b", "\x1b["):  # b, Esc
                            current_view = "dashboard"
                            current_detail = None
                            code_scroll = 0  # Reset scroll when leaving

                        # Toggle code
                        elif key == "c":
                            show_code = not show_code

                        # Scroll code down (j or down arrow)
                        elif key in ("j", "\x1b[B") and show_code and current_detail:
                            if current_detail.code:
                                total_lines = len(current_detail.code.split("\n"))
                                if code_scroll < total_lines - 10:
                                    code_scroll += 3

                        # Scroll code up (k or up arrow)
                        elif key in ("k", "\x1b[A") and show_code and current_detail:
                            code_scroll = max(0, code_scroll - 3)

                        # Refresh submission
                        elif key == "r" and current_detail:
                            submission_id = current_detail.submission.get("submission_id")
                            if submission_id:
                                current_detail = client.fetch_submission_detail(submission_id)

                # Auto-refresh dashboard
                if current_view == "dashboard" and time.time() - last_refresh > refresh_interval:
                    data = client.fetch_all()
                    last_refresh = time.time()
                    needs_redraw = True

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal
        print("\033[?1049l", end="", flush=True)  # Exit alternate screen buffer
        print("\033[?25h", end="", flush=True)  # Show cursor
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    """Main entry point for the TUI."""
    parser = argparse.ArgumentParser(
        prog="tplr",
        description="τemplar crusades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Dashboard:
    j/k (or ↑/↓)   Navigate entries
    h/l (or ←/→)   Switch between Leaderboard and Recent
    Enter/1-9      View submission details
    r              Refresh data
    q              Quit

  Submission Detail:
    c              Toggle code view
    r              Refresh
    b/Esc          Back to dashboard
    q              Quit

Examples:
  tplr                     # Auto-detect crusades.db if it exists
  tplr --demo              # Run with mock data for demo
  tplr --db crusades.db    # Read from validator's database
  tplr --url http://...    # Connect to API (legacy)
        """,
    )

    parser.add_argument(
        "--db",
        default="crusades.db",
        help="Path to validator's SQLite database (default: crusades.db)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Crusades API base URL (legacy mode)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Auto-refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with mock data for demonstration",
    )

    args = parser.parse_args()

    run_tui(args.url, args.refresh, demo=args.demo, db_path=args.db)


if __name__ == "__main__":
    main()
