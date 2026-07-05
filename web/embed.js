/*
 * yeGPT embed: self-contained, dependency-free, no backend, no build step.
 *
 * Exposes a global `yegptEmbed(containerElement, samplesUrl)` that fetches a
 * samples.json file (shape: {"samples": ["...", ...]}) and typewriter-streams
 * randomly chosen fragments into the container in an endless loop: type each
 * fragment character-by-character, pause, clear, pick the next. Nothing runs
 * live; the fragments are pregenerated model output replayed for effect.
 *
 * Returns a handle with `stop()` to cancel the loop.
 */
(function () {
  "use strict";

  var TYPE_DELAY_MS = 45;
  var HOLD_DELAY_MS = 1800;
  var CLEAR_DELAY_MS = 500;

  function sleep(ms) {
    return new Promise(function (resolve) {
      setTimeout(resolve, ms);
    });
  }

  function pickRandom(items) {
    return items[Math.floor(Math.random() * items.length)];
  }

  function fetchSamples(samplesUrl) {
    return fetch(samplesUrl).then(function (response) {
      if (!response.ok) {
        throw new Error("failed to fetch samples: " + response.status);
      }
      return response.json();
    }).then(function (data) {
      var samples = data && data.samples;
      if (!Array.isArray(samples)) {
        throw new Error("samples.json missing a 'samples' array");
      }
      return samples.filter(function (fragment) {
        return typeof fragment === "string" && fragment.length > 0;
      });
    });
  }

  function yegptEmbed(containerElement, samplesUrl) {
    if (!containerElement) {
      throw new Error("yegptEmbed: containerElement is required");
    }
    if (!samplesUrl) {
      throw new Error("yegptEmbed: samplesUrl is required");
    }

    var samples = [];
    var cancelled = false;

    containerElement.textContent = "";

    function typeFragment(text) {
      var index = 0;

      function step() {
        if (cancelled) {
          return Promise.resolve();
        }
        if (index >= text.length) {
          return Promise.resolve();
        }
        containerElement.textContent += text.charAt(index);
        index += 1;
        return sleep(TYPE_DELAY_MS).then(step);
      }

      containerElement.textContent = "";
      return step();
    }

    function loop() {
      if (cancelled) {
        return Promise.resolve();
      }
      if (samples.length === 0) {
        return sleep(1000).then(loop);
      }
      return typeFragment(pickRandom(samples))
        .then(function () {
          return sleep(HOLD_DELAY_MS);
        })
        .then(function () {
          if (!cancelled) {
            containerElement.textContent = "";
          }
          return sleep(CLEAR_DELAY_MS);
        })
        .then(loop);
    }

    fetchSamples(samplesUrl)
      .then(function (loaded) {
        samples = loaded;
      })
      .catch(function (error) {
        containerElement.textContent = "[samples unavailable]";
        if (typeof console !== "undefined") {
          console.error(error);
        }
      });

    loop();

    return {
      stop: function () {
        cancelled = true;
      }
    };
  }

  if (typeof window !== "undefined") {
    window.yegptEmbed = yegptEmbed;
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = yegptEmbed;
  }
})();
