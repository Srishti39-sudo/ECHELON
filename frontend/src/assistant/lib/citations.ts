/**
 * Citation markers, from the model's text to something clickable.
 *
 * The prompt asks for [S1] and, on a Mission Copilot answer, [D1], but
 * providers render citations their own way: fullwidth brackets, parentheses,
 * and runs collapsed into [S1, S3, D2]. The backend's grounding check handles
 * all of them, and so must this, or a cited answer renders as plain text with
 * brackets in it.
 *
 * The rewrite turns every marker into a markdown link with a `#cite-N` (source)
 * or `#data-N` (survey record) target, which the renderer maps to a clickable
 * marker component. Doing it before parsing keeps the rest of the markdown
 * untouched.
 */

const GROUP = /[[(【]([^[\]()【】]{0,80})[\])】]/g
// S matches as the backend's CITATION_REF_PATTERN does; D needs a boundary so
// an identifier such as "ID12" is not read as a data citation.
const REF = /((?<![A-Za-z0-9_])D|S)\s*(\d+)/g

export type CitationKind = 'source' | 'data'

export interface CitationTarget {
  kind: CitationKind
  n: number
}

export function linkCitations(markdown: string): string {
  return markdown.replace(GROUP, (whole, inner: string) => {
    const refs = Array.from(inner.matchAll(REF), (m) => ({ kind: m[1], n: Number(m[2]) }))
    if (refs.length === 0) return whole
    return refs
      .map((ref) =>
        ref.kind === 'D' ? `[D${ref.n}](#data-${ref.n})` : `[${ref.n}](#cite-${ref.n})`,
      )
      .join('')
  })
}

/** The citation behind a `#cite-N` or `#data-N` href, or null for an ordinary link. */
export function citationTarget(href: string | undefined): CitationTarget | null {
  if (!href) return null
  const match = /^#(cite|data)-(\d+)$/.exec(href)
  if (!match) return null
  return { kind: match[1] === 'data' ? 'data' : 'source', n: Number(match[2]) }
}
