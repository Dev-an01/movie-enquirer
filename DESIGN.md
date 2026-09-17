---
name: Movie Enquirer Showcase
description: A clean, accessible Streamlit UI for inspecting a local RAG movie search engine.
colors:
  primary:
    value: "#2563eb"
  secondary:
    value: "#1E3A8A"
  neutral:
    value: "#f9fafb"
  text:
    value: "#1f2937"
  subtle:
    value: "#4B5563"
  success:
    value: "#059669"
typography:
  header:
    family: "system-ui, sans-serif"
    weight: 700
    size: "2.5rem"
  title:
    family: "system-ui, sans-serif"
    weight: 700
    size: "1.25rem"
  body:
    family: "system-ui, sans-serif"
    weight: 400
    size: "1rem"
rounded:
  md: 8px
  lg: 10px
spacing:
  sm: "0.5rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2rem"
---

# Overview
The Movie Enquirer interface prioritizes clarity, performance, and immediate understanding of the retrieval results. It is an unopinionated container for a 5,000-record movie corpus, pairing a straightforward layout with elevated typography and distinct "cards" for each result. The UI avoids unnecessary decorative elements in favor of a content-first, data-rich presentation.

**Key Characteristics:**
- Content-first layout centered on readable movie cards
- Clear hierarchy separating titles, scores, and descriptions
- Functional, unobtrusive imagery to aid scanning

# Colors
The palette is built around trust and readability, using muted grays for structure and deep blues for identity and interaction.

- **Brand Blue** (#1E3A8A): Used for the main header to establish the identity.
- **Action Blue** (#2563eb): Used for primary buttons and the left border of movie cards.
- **Surface Gray** (#f9fafb): The background color of movie cards, lifting them off the canvas.
- **Primary Text** (#1f2937): High-contrast dark gray for primary reading text.
- **Secondary Text** (#4B5563): Muted gray for subheaders and less critical information.
- **Success Green** (#059669): Used specifically for highlighting the `rrf_score`.

# Typography
Typography relies on the native `system-ui` stack to ensure fast loading, familiar legibility, and no external dependencies.

- **Main Header**: Large and bold (2.5rem, 700) to anchor the page.
- **Movie Titles**: Distinctly bold (1.25rem, 700) to stand out within the cards.
- **Body Text**: Standard readability scale (1rem) for descriptions.

# Layout
The UI uses a fluid, wide layout (`layout="wide"`) with a persistent sidebar for global navigation and context, and a main content area divided into functional tabs.

**The Proximity Rule.** Card contents are tightly grouped, while a healthy margin (1rem) separates distinct cards from each other.

# Elevation & Depth
Depth is used sparingly to define interactive or distinct elements.

- **Cards**: Movie cards use a subtle drop shadow (`0 4px 6px -1px rgba(0, 0, 0, 0.1)`) and a distinct left border to establish them as contained, discrete units of information.
- **Images**: Movie posters feature a softer shadow (`0 2px 4px rgba(0,0,0,0.1)`) to separate them from the card surface.

# Shapes
Corners are moderately rounded to create a friendly, modern aesthetic without feeling overly playful.

- **Cards**: Large radius (10px).
- **Images**: Medium radius (8px).

# Components
## Movie Card
The fundamental building block of the search results.

- **Structure**: A flex container with a 120px fixed-width image on the left, and a flexible text column on the right.
- **Styling**: `#f9fafb` background, 10px rounded corners, subtle shadow, and a 5px solid `#2563eb` left border.
- **Content**: Always displays the rank, title, relevance score (in green), and description. Fails gracefully if the image link is missing.

# Do's and Don'ts
- **Do** rely on standard Streamlit components for inputs, buttons, and tabs to ensure native performance and accessibility.
- **Do** use custom HTML/CSS for the movie cards to achieve a specific layout that standard Streamlit components cannot easily replicate.
- **Don't** add complex interactions or JavaScript-heavy elements; the UI must remain a plain, inspectable showcase.
- **Don't** use color alone to convey meaning, except as an accent for the relevance score.
