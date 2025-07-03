# UUBED Documentation Site

This directory contains the Jekyll-based documentation for UUBED, configured to work with GitHub Pages.

## Local Development

To run the documentation site locally:

1. Install Ruby and Bundler:
   ```bash
   gem install bundler
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Run the Jekyll server:
   ```bash
   bundle exec jekyll serve --baseurl ""
   ```

4. Open http://localhost:4000 in your browser

## GitHub Pages Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the main branch. The site will be available at:

https://twardoch.github.io/uubed-docs/

## Structure

- `_config.yml` - Jekyll configuration
- `_layouts/` - Page templates
- `_includes/` - Reusable components
- `assets/css/` - Custom stylesheets
- `index.md` - Home page
- Other `.md` files - Documentation pages

## Adding New Pages

1. Create a new `.md` file
2. Add Jekyll front matter:
   ```yaml
   ---
   layout: page
   title: Your Page Title
   description: Brief description
   ---
   ```
3. Write your content in Markdown

## Theme

The site uses the Cayman theme, which is built into GitHub Pages.