# Data: Ferdowsi's *Shahnameh*

`shahnameh.txt` is the complete *Shahnameh* (شاهنامه, "Book of Kings") of
Abu'l-Qasim Ferdowsi, the Persian national epic, completed around 1010 CE.

| file | content |
|---|---|
| `shahnameh.txt` | one couplet (*bayt*) per line, the two hemistichs (*mesra*) separated by ` / `; sections separated by a blank line |
| `sections.json` | for every section: Ganjoor id, story, title, first line, number of couplets |

**Source.** Downloaded from the [Ganjoor](https://ganjoor.net) digital library of
Persian poetry through its public API, using `scripts/fetch_shahnameh.py`
(section titles, which are editorial, are not included in the text). Ganjoor's
own annotations (summaries, comments) are discarded; only the verse itself is kept.

**Licence.** Ferdowsi's poem is about a thousand years old and in the public
domain. The digital transcription is Ganjoor's; it is used here for
non-commercial research, with attribution, as is common practice for Persian
NLP corpora. If you reuse the text, please credit Ganjoor.

**Why the Persian original and not an English translation?** See
[`docs/DESIGN.md`](../docs/DESIGN.md#1-dataset). In short: the original is
verse with a checkable form (every couplet rhymes), it is unambiguously public
domain, and the most famous public-domain English version (Helen Zimmern's
1883 *Epic of Kings*) is an abridged prose retelling, which would make "the
model writes verse" untrue.

**Normalisation.** Whitespace is collapsed; the zero-width non-joiner (U+200C),
which is part of Persian spelling (e.g. می‌رود), is kept. Nothing else is changed.
