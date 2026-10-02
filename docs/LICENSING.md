# Licensing

## AGPL-3.0

`LICENSE` carries gpark's copyright notice, followed by the **verbatim** GNU Affero
General Public License version 3. The licence text is unmodified; the provenance
below lives here instead, so the licence document itself stays byte-identical to the
FSF's.

### Provenance

| | |
|---|---|
| Source | `https://www.gnu.org/licenses/agpl-3.0.txt` |
| Retrieved | 2026-10-02 |
| Size | 661 lines, 34 523 bytes |
| SHA-256 | `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` |

To verify that the block in `LICENSE` is still unmodified:

```sh
tail -c 34523 LICENSE | shasum -a 256
# 0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0
```

Or re-fetch and compare directly:

```sh
curl -fsSL https://www.gnu.org/licenses/agpl-3.0.txt -o /tmp/agpl.txt
cmp <(tail -c "$(wc -c < /tmp/agpl.txt)" LICENSE) /tmp/agpl.txt && echo identical
```

If the FSF ever revises the text, the checksum will no longer match. That is a prompt
to re-fetch deliberately, not an error to paper over — silently updating a licence
file is how provenance gets lost.

### Why AGPL

The network clause is the point, not a side effect. It guarantees that anyone who
modifies gpark publishes their changes, which is what protects the direct-PTX route
into every backend added later.

That protection only means something if the abstraction is genuinely neutral across
backends, so it pairs with the cross-backend parity gate in `ROADMAP.md`: a backend
that cannot do an operation has to fail loudly rather than silently emit a slower
fallback. Taichi is the cautionary case — nominally neutral on three vendors, and
genuinely so on one. See `TAICHI-NOTES.md`.

Decision record: `DECISIONS.md`.

### Outstanding

Nothing. This file previously carried only the notice and an instruction for a
maintainer to fetch the text; that gap is now closed.
