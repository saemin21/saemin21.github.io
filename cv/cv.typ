// CV PDF, built from the same data as the /cv/ page.
// Build: make cv   (passes the publication years found in data/publications/)

#let load(name) = yaml("/data/cv/" + name + ".yaml")
#let cv = (
  ..load("profile"),
  education: load("education"),
  interests: load("interests"),
  awards: load("awards"),
  experience: load("experience"),
  skills: load("skills"),
)
// Typst cannot list a directory, so the years come in as --input pub_years=2026,2025,...
#let pub_years = sys.inputs.at("pub_years", default: "").split(",").filter(y => y != "")
#let publications = (
  pub_years
    .sorted()
    .rev()
    .map(y => yaml("/data/publications/" + y + ".yaml").map(p => p + (year: y)))
    .flatten()
)

#set document(title: cv.name + " – CV", author: cv.name)
#set page(paper: "a4", margin: (x: 2.2cm, y: 2cm))
#set text(font: "Libertinus Serif", size: 10.5pt)
#set par(leading: 0.62em)
#set list(indent: 1em, spacing: 0.75em)
#set enum(indent: 1em, spacing: 0.75em)
#show link: set text(fill: rgb("#1f4e8c"))

// sticky: a heading never stays alone at the bottom of a page
#let section(title) = block(sticky: true, above: 1.6em, below: 0.7em)[
  #text(size: 12.5pt, weight: "bold", smallcaps(title))
  #v(-0.75em)
  #line(length: 100%, stroke: 0.6pt)
]

#let small(body) = text(size: 9.5pt, fill: luma(60), body)

#let authors(p) = {
  let eq = p.at("equal", default: 0)
  let n = p.authors.len()
  let names = p.authors.enumerate().map(((i, a)) => {
    let s = if i < eq { a + "*" } else { a }
    if a == cv.name { strong(s) } else { s }
  })
  // very long lists (e.g. team papers): first 3, me, and the count
  if n > 8 {
    let me = p.authors.position(a => a == cv.name)
    let mine = if me != none { (names.at(me),) } else { () }
    names = names.slice(0, 3) + ("…",) + mine + ([… (#n authors)],)
  }
  names.join(", ")
}

#let pub(p) = [
  #authors(p), "#p.title", #emph(p.venue), #p.year
  #if "awards" in p [ \ #small[Award: #p.awards.join("; ")]]
  #if "workshops" in p [ \ #small[Also presented at #p.workshops.join("; ")]]
  #if p.at("equal", default: 0) > 0 [ \ #small[\*: equal contribution]]
]

// ---------------------------------------------------------------- header
#align(center)[
  #text(size: 22pt, weight: "bold", cv.name)
  #v(0.2em)
  #cv.position \
  #cv.affiliation \
  #cv.address \
  #link("mailto:" + cv.email, cv.email) \
  #link("https://" + cv.website, cv.website)
]

// ---------------------------------------------------------------- body
#section("Education")
#for e in cv.education {
  grid(
    columns: (2.7cm, 1fr, auto),
    column-gutter: 0.8em,
    [#e.start \~ \ #e.end],
    [*#e.school* \ #e.department #for d in e.details [ \ #emph(d)]],
    align(right, e.location),
  )
  v(0.6em)
}

#section("Research Interests")
#list(..cv.interests)

#section("Publications")
#enum(..publications.map(pub))

#section("Awards and Honors")
#list(..cv.awards.map(a => [
  #a.title, #a.location (#a.date)#if "detail" in a [ \ #small(a.detail)]
]))

#section("Research Experience")
#list(..cv.experience.map(x => [
  #x.role (#x.period)#if "detail" in x [ \ #small(x.detail)]
]))

#section("Skills")
#list(..cv.skills)

#v(1fr)
#align(right, small[Last updated: #datetime.today().display("[month repr:long] [year]")])
