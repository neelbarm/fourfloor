"""Reference ingest: turn a pile of remix links into pairs fourfloor can learn from.

A DJ's taste lives in the remixes he saves, and every one of them is half of a
lesson: the other half is the record it was made from. ``fourfloor refs`` takes
the link, works out what the original was, *proves* the two are the same song by
listening to both, files the pair under the convention ``fourfloor learn``
already reads, samples a drum kit out of the remix, and folds what it measured
back into the defaults.

The modules, in the order the pipeline uses them:

* :mod:`~fourfloor.refs.titles` -- what a remix title says: artist, track,
  remixer, and the junk in between.
* :mod:`~fourfloor.refs.search` -- ask yt-dlp for candidate originals and rank
  them by title similarity, plausible duration and the words that give away
  another edit.
* :mod:`~fourfloor.refs.verify` -- the part that matters: are these two files
  the same song? Vocal chroma, every tempo and pitch relation a remixer might
  have used, subsequence DTW, one score.
* :mod:`~fourfloor.refs.ledger` -- a resumable record of what was done, so a
  hundred pasted links survive a closed laptop.
* :mod:`~fourfloor.refs.pipeline` -- the whole run, one link at a time.
* :mod:`~fourfloor.refs.learned` -- the table the pairs teach, and the two
  engine decisions that read it.

Audio never enters the repository. Pairs live in ``~/Music/house-refs``, kits in
``~/.fourfloor/kits``, and the only thing written back into the project is a set
of aggregate numbers with no filenames and no titles in it.
"""

from __future__ import annotations

REFS_SCHEMA = 1
