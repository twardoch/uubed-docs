---
layout: default
title: "Documentation README"
exclude: true
description: "Development documentation for the UUBED documentation site"
---

> This is the main hub for the UUBED documentation website. It's built with Jekyll and designed for GitHub Pages, making it easy to develop locally and deploy automatically. Think of it as the instruction manual for building the instruction manual!

# UUBED Documentation Site

Imagine this directory is the blueprint for a magnificent library. Every file here is a detailed instruction, from how the shelves are built (`_layouts/`) to where the books are stored (`_config.yml`). It's all designed to make sure our knowledge is beautifully presented and easily accessible.

Imagine you're a master chef, and this directory is your recipe book for creating the perfect documentation feast. Each section tells you how to prepare the ingredients, cook the dishes, and even how to serve them up for the world to enjoy.

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