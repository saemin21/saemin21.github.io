# Local helpers. CI (.github/workflows/hugo.yml) runs the same steps on push to main.

PUB_YEARS := $(shell ls data/publications | sed -n 's/\.yaml$$//p' | paste -sd, -)
TYPST := typst compile --root . --ignore-system-fonts --input pub_years=$(PUB_YEARS)

.PHONY: cv cv-watch serve build

cv:
	$(TYPST) cv/cv.typ static/cv.pdf

cv-watch:
	typst watch --root . --ignore-system-fonts --input pub_years=$(PUB_YEARS) cv/cv.typ static/cv.pdf

serve: cv
	hugo server

build: cv
	hugo --gc --minify
