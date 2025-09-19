This is a comprehensive guide to reveal.js, designed to help you create and customize your presentations.

1. Getting Started

This section covers the initial setup, including installation and basic HTML structure.

1.1 Installation

You can get reveal.js by downloading a pre-built version from GitHub or by installing it from npm.

NPM:

Generated bash
npm install reveal.js


Manual Download:
Download and unzip the library from the official GitHub repository.

1.2 Basic HTML Setup

Include the core CSS and JS files in your HTML. The basic structure consists of a .reveal container with a .slides wrapper for your <section> elements.

Generated html
<!doctype html>
<html>
<head>
  <link rel="stylesheet" href="dist/reveal.css">
  <link rel="stylesheet" href="dist/theme/black.css">
</head>
<body>
  <div class="reveal">
    <div class="slides">
      <section>Slide 1</section>
      <section>
        <!-- Nested sections create vertical slides -->
        <section>Vertical Slide 1</section>
        <section>Vertical Slide 2</section>
      </section>
      <section>Slide 3</section>
    </div>
  </div>
  <script src="dist/reveal.js"></script>
  <script>
    // Initialize Reveal
    Reveal.initialize({
      // Configuration options go here
    });
  </script>
</body>
</html>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
2. Content & Media

Add various types of content to your slides, from Markdown to background videos.

2.1 Markdown Support

You can write your slide content in Markdown, which is often faster and cleaner than HTML.

1. Enable Plugin:

Generated html
<script src="plugin/markdown/markdown.js"></script>
<script>
  Reveal.initialize({
    plugins: [ RevealMarkdown ]
  });
</script>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

2. Link External Markdown File:

Generated html
<section data-markdown="slides.md"
         data-separator="^\r?\n---\r?\n$"
         data-separator-vertical="^\r?\n--\r?\n$"
         data-separator-notes="^Note:">
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
2.2 Backgrounds

You can set a background for the entire presentation or for individual slides.

Feature	Attribute	Example
Color	data-background-color	data-background-color="tomato"
Image	data-background-image	data-background-image="image.png"
Video	data-background-video	data-background-video="video.mp4"
Iframe	data-background-iframe	data-background-iframe="https://example.com"

Example with advanced options:

Generated html
<section data-background-color="#222"
         data-background-image="background.jpg"
         data-background-size="cover"
         data-background-position="center"
         data-background-repeat="no-repeat">
  <h2>Slide with a Background</h2>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
2.3 Code Highlighting

Display code with syntax highlighting using the highlight.js plugin.

1. Enable Plugin & Theme:

Generated html
<link rel="stylesheet" href="plugin/highlight/monokai.css">
<script src="plugin/highlight/highlight.js"></script>
<script>
  Reveal.initialize({
    plugins: [ RevealHighlight ]
  });
</script>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

2. Usage:
Use <pre><code> blocks. The data-trim attribute removes leading/trailing whitespace, and data-line-numbers adds line numbering.

Generated html
<section>
  <pre><code data-trim data-line-numbers class="hljs javascript">
function greet() {
  console.log("Hello, reveal.js!");
}
  </code></pre>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
2.4 Math Equations (KaTeX)

Render mathematical formulas using the Math plugin.

1. Enable Plugin:

Generated html
<script src="plugin/math/math.js"></script>
<script>
  Reveal.initialize({
    plugins: [ RevealMath.KaTeX ]
  });
</script>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

2. Usage:
Wrap your LaTeX math in \(...\) for inline and \[...\] for block equations.

Generated html
<section>
  <h2>The Quadratic Formula</h2>
  <p>
    \[ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]
  </p>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
2.5 Media Elements

You can embed images, videos, audio, and iframes directly into your slides.

Generated html
<section>
  <img src="diagram.png" alt="Diagram" width="500">
  <video controls autoplay loop src="animation.mp4"></video>
  <audio controls src="music.mp3"></audio>
  <iframe src="https://example.com" width="800" height="400"></iframe>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
3. Layout & Navigation

Control the look, feel, and flow of your presentation.

3.1 Themes & Sizing

Set the visual theme and dimensions of your presentation. Themes are included in dist/theme/.

Generated html
<!-- Include a theme in the <head> -->
<link rel="stylesheet" href="dist/theme/league.css">
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
Generated javascript
// Set dimensions and scaling in the initialize call
Reveal.initialize({
  width: 1280,
  height: 720,
  margin: 0.04,
  center: true, // Vertically center slide content
  minScale: 0.2,
  maxScale: 2.0
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END
3.2 Transitions

Animate the transition between slides.

Generated javascript
Reveal.initialize({
  transition: 'convex',        // none, fade, slide, convex, concave, zoom
  transitionSpeed: 'default',  // default, fast, slow
  backgroundTransition: 'fade' // none, fade, slide, convex, concave, zoom
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END

You can also override this for specific slides:

Generated html
<section data-transition="zoom">This slide will zoom in.</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
3.3 Auto-Animate

Automatically animate matching elements between slides. Elements are matched by their text content, src attribute, or a data-id attribute.

Generated html
<!-- Slide 1 -->
<section data-auto-animate>
  <h1 style="color: blue;">reveal.js</h1>
</section>

<!-- Slide 2 -->
<section data-auto-animate>
  <h1 style="color: red; margin-top: 100px;">reveal.js</h1>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

This will animate the <h1> element from its position and color on the first slide to the new values on the second.

3.4 Fragments

Reveal content incrementally on a single slide.

Generated html
<section>
  <p class="fragment">First Step</p>
  <p class="fragment fade-up">Second Step</p>
  <p class="fragment highlight-red" data-fragment-index="3">Third Step</p>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

Common fragment styles include fade-in, fade-out, fade-up, zoom-in, and highlight-red.

3.5 Internal Links

Link between slides using the slide's index or an ID.

Generated html
<a href="#/2/1">Go to slide H:2, V:1</a>
<a href="#/ending">Go to the 'ending' slide</a>

<!-- ... -->

<section id="ending">
  <h2>The End</h2>
</section>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
3.6 Slide Numbers and Progress

Display the slide number and a progress bar.

Generated javascript
Reveal.initialize({
  slideNumber: 'c/t', // c, c/t, h.v, h/v
  progress: true,     // Display the progress bar
  showSlideNumber: 'all' // all, print, speaker
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END
4. Features & Plugins

Extend reveal.js with powerful features like speaker notes, search, and more.

4.1 Speaker Notes

Add notes that are only visible in the speaker view (press S to open).

1. Enable Plugin:

Generated html
<script src="plugin/notes/notes.js"></script>
<script>
  Reveal.initialize({ plugins: [ RevealNotes ] });
</script>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END

2. Usage:

Generated html
<section>
  <h2>A Slide Title</h2>
  <aside class="notes">
    These are my speaker notes. They won't be visible on the main screen.
  </aside>
</section>

<!-- Or use the shorthand attribute -->
<section data-notes="This is a quick note."></section>```

#### 4.2 PDF Export

Generate a PDF version of your presentation by appending `?print-pdf` to the URL and using your browser's print function. For best results, use Chrome/Chromium and save to PDF with "Background graphics" enabled and margins set to "None".

#### 4.3 Fullscreen & Overview

- **Fullscreen:** Press `F` to enter or exit fullscreen mode.
- **Overview:** Press `Esc` or `O` to toggle the slide overview, which shows a birds-eye view of your presentation.

#### 4.4 Additional Built-in Plugins

| Plugin | Path | Description |
|---|---|---|
| **RevealSearch** | `plugin/search/search.js` | Adds a search box (Ctrl+Shift+F). |
| **RevealZoom** | `plugin/zoom/zoom.js` | Lets you zoom into parts of a slide (Alt+Click). |
| **RevealChalkboard** | `plugin/chalkboard/chalkboard.js` | Provides a digital chalkboard for annotations. |
| **RevealMenu** | `plugin/menu/menu.js` | Adds a customizable slide-out menu. |

To use them, add them to your `plugins` array in the `Reveal.initialize` config.

### 5. Advanced Control

Customize the behavior of your presentation with the API, custom keybindings, and automated playback.

#### 5.1 JavaScript API

You can control the presentation programmatically.

```javascript
// Wait for Reveal to be ready
Reveal.initialize({ /* config */ }).then(() => {
  // Your code here, e.g., binding to events
});

// Navigate to slide H:2, V:1, Fragment:0
Reveal.slide(2, 1, 0);

// Move to the next slide or fragment
Reveal.next();

// Get the current slide indices {h, v, f}
let indices = Reveal.getIndices();

// Re-apply layout after DOM changes
Reveal.layout();

// Update configuration at runtime
Reveal.configure({ loop: true });
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Html
IGNORE_WHEN_COPYING_END
5.2 Events

Listen for events to trigger custom behavior.

Generated javascript
Reveal.on('ready', event => {
  console.log('Reveal is ready.');
});

Reveal.on('slidechanged', event => {
  // event.previousSlide, event.currentSlide, event.indexh, event.indexv
  console.log(`Slide changed to ${event.indexh}, ${event.indexv}`);
});

Reveal.on('fragmentshown', event => {
  // event.fragment = the DOM element of the fragment
  console.log('A fragment was shown.');
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END
5.3 Auto-Slide Playback

Automatically cycle through slides.

Generated javascript
Reveal.initialize({
  autoSlide: 5000, // Advance every 5 seconds
  loop: true,      // Loop back to the beginning
  autoSlideStoppable: true // Pause when the user navigates
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END

You can override the timing per-slide or per-fragment with data-autoslide="<milliseconds>".

5.4 Keyboard Bindings

Customize or add new keyboard shortcuts.

Generated javascript
Reveal.initialize({
  keyboard: {
    13: 'next', // Go to the next slide when Enter is pressed
    32: null    // Disable the default spacebar action
  }
});

// Add a new custom key binding
Reveal.addKeyBinding({ keyCode: 84, key: 'T', description: 'Show Timer' }, () => {
  console.log('T was pressed!');
});
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END