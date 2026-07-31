# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`
- **Read an issue**: `gh issue view <number> --comments`, including labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments` with appropriate label and state filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`
- **Close an issue**: `gh issue close <number> --comment "..."`

Infer the repository from `git remote -v`; `gh` does this automatically when run inside this clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

When set to `yes`, pull requests run through the same labels and states as issues using the corresponding `gh pr` commands.

GitHub shares one number space across issues and pull requests. A bare `#42` may refer to either; resolve it with `gh pr view 42` and fall back to `gh issue view 42`.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: An issue labelled `wayfinder:map`, containing the Notes, Decisions-so-far, and Fog sections.
- **Child ticket**: An issue linked to the map as a GitHub sub-issue. If sub-issues are unavailable, add it to a task list in the map and put `Part of #<map>` at the top of the child body.
- **Ticket labels**: Use `wayfinder:<type>`, where type is `research`, `prototype`, `grilling`, or `task`.
- **Blocking**: Prefer GitHub's native issue dependencies. If unavailable, use a `Blocked by: #<n>, #<n>` line at the top of the child body.
- **Frontier**: Find the first open, unblocked, and unassigned child ticket in map order.
- **Claim**: Assign the ticket to the current developer before beginning work.
- **Resolve**: Comment with the answer, close the ticket, and append a context pointer to the map's Decisions-so-far section.
