# DeepEcho knowledge base

Every document the assistant is allowed to answer from lives here. It answers
from nothing else.

## Status field

Each file carries front matter with a `status`:

- `verified` - written from a real publication, which is named in `source_file`
  and kept in `sources/`.
- `PLACEHOLDER` - seed content written to make the pipeline runnable. Not a
  source. `rag.py index` warns on every one, and the assistant is under
  instruction to flag any citation to one.

Every file in this directory is `verified`. Each names the publication it came
from, and `sources/PROVENANCE.md` records where that publication was obtained
and when.

## What the corpus still does not contain

These are gaps in the published record, not gaps in the pipeline, and the
assistant reports them rather than filling them.

| Missing | Where it bites |
|---|---|
| A universal numeric standoff distance | Most sources decline to give one without the explosive weight |
| An Indian authority for an ordnance report | The only Indian channel here is pollution reporting, which must not be offered for ordnance |
| Seabed-specific debris categories | NOAA's categorisation guide is a shoreline survey instrument |

The single most valuable document to add is an Indian reporting procedure for
suspected ordnance, from the Indian Navy or the Indian Coast Guard.

## Adding a document

Drop any `.md` or `.txt` file in here with front matter, then re-index. Chunking
splits on `##` headings, so keep one topic per heading.
