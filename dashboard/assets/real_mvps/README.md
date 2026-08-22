# Real MVP photos (optional)

This folder is empty by default -- the project ships with no real player
photos (licensing real NBA player photos requires rights this project
doesn't have).

To add your own licensed images, drop a file here named `<year>.<ext>`
(`.jpg`, `.jpeg`, `.png`, or `.webp`), where `<year>` is the season-ending
year -- e.g. `2016.jpg` for the 2015-16 MVP (Stephen Curry). Run
`python -m dashboard.build` again afterward and it'll be picked up
automatically by `dashboard/real_history.py:real_mvp_image_data_uri`.

Only use images you have the rights to redistribute.
