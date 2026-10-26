# saemin21.github.io

Personal homepage and blog, built with [Hugo](https://gohugo.io) + [PaperMod](https://github.com/adityatelange/hugo-PaperMod) and deployed to GitHub Pages by `.github/workflows/hugo.yml`.

The previous Jekyll (Chirpy) site is kept on the `legacy` branch.

## What to edit

| To change | Edit |
|---|---|
| Home intro | `content/_index.md` |
| Home news | `data/news.yaml` |
| Publications | `data/publications/<year>.yaml` (one file per year; new year = new file) |
| CV header (name, email, …) | `data/cv/profile.yaml` |
| Education / interests / awards / experience / skills | `data/cv/<section>.yaml` |
| Menu, social icons | `hugo.yaml` |
| Profile photo | `assets/img/profile.jpg` |
| Styles (badge colors, …) | `assets/css/extended/custom.css` |

The publications page, the CV page (`/cv/`) and the CV PDF (`/cv.pdf`, built by `cv/cv.typ`) all read the same files.
Blog posts in `content/posts/` are hidden for now (see `content/posts/_index.md`).

## Local

```bash
git submodule update --init   # PaperMod theme
brew install hugo typst
make serve                    # builds cv.pdf, then runs hugo server
make cv-watch                 # (optional, 2nd terminal) rebuild cv.pdf on every save
```

