# Labeling guide

Label files are the ground truth the system is measured against. Fill them in from the transcript
alone, **before** looking at what the system produced, and apply the rules below the same way in
every meeting. If you disagree with a rule, change it here and re-check the labels you already wrote.

## Files

| File | Used for |
|---|---|
| `meetings_dev.jsonl`, `decider_dev.jsonl` | Tuning prompts, rules and thresholds. |
| `meetings_test.jsonl`, `decider_test.jsonl` | Reporting final results. Never tune against these. |

Every row has a `status`:

- `template`: empty skeleton, ignored by the evaluations.
- `draft`: filled in but not verified. Only used with `--allow-draft`.
- `reviewed`: checked by a person. This is the only status that counts as ground truth.

## Meeting labels

- `tasks`: one entry per task the system should find.
  - `evidence_quote`: a sentence copied from the transcript that proves the task. A predicted task is
    recognised by this quote, so it must come from the transcript, not be reworded.
  - `assignee`: first name, or `null` if the transcript does not say who.
  - `due_date`: `YYYY-MM-DD`, or `null`.
  - `commitment`: `firm`, or `conditional` ("if I have time", "if you want").
- `not_task_quotes`: sentences that sound like tasks but are not (vague ideas, "we should...", small talk).
  The system is penalised if it turns one of them into a task.

## Rules

1. **A task is something a person commits to do, or is explicitly asked to do and accepts.**
   Ideas and wishes nobody takes on ("we should think about...") are not tasks.
2. **Conditional commitments are tasks**, labelled `commitment: conditional`.
3. **An action completed on the spot** ("Schedule a call." "Done, scheduled.") is not a task.
4. **One task per person.** If two people take on the same work, label two tasks.
5. **Assignee** is the person who will do the work, not the one who asked.
6. **Dates are resolved from the meeting date.** "Tomorrow" is the next day; "Friday" is the next
   Friday. "This afternoon", "today" and "after lunch" mean the meeting date.
7. **"Before <weekday>"** means that weekday ("before Thursday at noon" is Thursday).
8. **A deadline given in a request applies to whoever accepts it**, even if they do not repeat it.
9. **No date stated or implied** means `null`; never guess one.

## Fragment labels (Decider)

Each row is one sentence. Answer as if that sentence were the only evidence for a task.

- `statement_type`: `task`, `decision`, `suggestion`, `risk`, `question` or `other`.
- `explicit_commitment`: `true` if someone clearly commits to doing something, or is clearly asked
  to. `false` for conditional offers and optional requests.
- `evidence_sufficient`: `true` if the sentence alone justifies creating a task.
- `note`: write `borderline` when a reasonable person could answer differently. Those rows show
  where the labels, not the system, are uncertain.

## Reviewing a draft

Read each row against the transcript. Fix what is wrong, then change `status` to `reviewed`.
