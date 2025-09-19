console.log("Initializing WebSocket connection...");

function connectWebSocket() {
  // Connect to the dedicated WebSocket server
  const wsUrl = "ws://localhost:8765";
  const ws = new WebSocket(wsUrl);
  console.log("Attempting connection to:", wsUrl);

  ws.onopen = function(event) {
    console.log("WebSocket connection established successfully");
  };

  ws.onerror = function(error) {
    console.error("WebSocket error:", error);
  };

  ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log("Received event:", data.event);

    const toolbar = document.getElementById('tutorial-toolbar');
    const welcomeText = document.getElementById('instruction-welcome');
    const unlockText = document.getElementById('instruction-unlock');
    const swipeText = document.getElementById('instruction-swipe');
    const flipText = document.getElementById('instruction-flip');
    const idleTooltip = document.getElementById('idle-tooltip');

    const idleHints = [
        "Try swiping your hand left to go back.",
        "You can lock the system by performing the 'rolling' gesture again.",
        "Remember, some slides have content below them. Use the 'flip' gesture to see!",
        "Just try out some gestures!",
        "Did you know you can navigate up, down, left, and right?",
        "Lost? Use the 'rotate' gesture to reset the presentation to the beginning."
    ];

    // --- Helper to hide all instructions ---
    function hideAllInstructions() {
        welcomeText.classList.remove('visible');
        unlockText.classList.remove('visible');
        swipeText.classList.remove('visible');
        flipText.classList.remove('visible');
        idleTooltip.classList.remove('visible');
    }

    switch (data.event) {
      case "user_detected":
        toolbar.classList.add('visible');
        welcomeText.innerText = "Hey you. Yes you, looking at the screen.";
        welcomeText.classList.add('visible');
        setTimeout(() => {
            if (welcomeText.classList.contains('visible')) {
                welcomeText.classList.remove('visible');
                unlockText.innerText = "To begin, unlock the system by rolling your arms in a circle.";
                unlockText.classList.add('visible');
            }
        }, 4000);
        break;

      case "show_idle_tooltip":
        const randomIndex = Math.floor(Math.random() * idleHints.length);
        idleTooltip.innerText = idleHints[randomIndex];
        idleTooltip.classList.add('visible');
        break;

      case "hide_idle_tooltip":
        idleTooltip.classList.remove('visible');
        break;

      case "system_reset":
        document.body.classList.add('locked');
        hideAllInstructions();
        toolbar.classList.remove('visible');
        Reveal.slide(0, 0); // Go to the first slide
        break;

      case "system_locked":
        document.body.classList.add('locked');
        break;

      case "system_unlocked":
        document.body.classList.remove('locked');
        document.body.classList.add('pulse-animation');
        setTimeout(() => document.body.classList.remove('pulse-animation'), 1000);
        
        if (unlockText.classList.contains('visible')) {
            unlockText.classList.remove('visible');
            swipeText.innerText = "Great! System unlocked. Now, swipe your hand from right to left to continue.";
            swipeText.classList.add('visible');
        }
        break;

      case "right":
        if (toolbar.classList.contains('visible')) {
            hideAllInstructions();
            // Show the next instruction after a short delay
            setTimeout(() => {
                flipText.innerText = "You can also navigate down. Try the 'flip' gesture.";
                flipText.classList.add('visible');
            }, 800);
        }
        Reveal.right();
        break;

      case "left":
        hideAllInstructions();
        Reveal.left();
        break;
        
      case "down":
        hideAllInstructions();
        toolbar.classList.remove('visible');
        Reveal.down();
        break;

      default:
        console.debug(`Unknown event received: ${data.event}`);
    }
  };

  ws.onclose = function(event) {
    console.log("WebSocket connection closed with code:", event.code);
    console.log("Close reason:", event.reason || "No reason provided");
    console.log("Clean closure:", event.wasClean);

    setTimeout(function() {
      console.log("Attempting to reconnect...");
      socket = connectWebSocket();
    }, 5000);
  };

  return ws;
}

// The WebSocket connection is now initialized in slideshow.html
// after the DOM is fully loaded.
// let socket = connectWebSocket();

// --- Hide tooltip on any user interaction ---
function hideTooltipOnAction() {
    const idleTooltip = document.getElementById('idle-tooltip');
    if (idleTooltip && idleTooltip.classList.contains('visible')) {
        idleTooltip.classList.remove('visible');
    }
}

document.addEventListener('keydown', hideTooltipOnAction);
document.addEventListener('click', hideTooltipOnAction);
