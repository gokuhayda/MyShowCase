# Assets

This folder contains visual assets for the UGFT Simulator.

## Required Files

Before deployment, add the following files:

### `preview.gif`
- Animated preview of the simulator in action
- Recommended: 600x400px, 10-15 seconds loop
- Shows phase synchronization emerging

### `screenshot.png`
- Static screenshot for documentation
- Recommended: 1200x800px
- Shows full interface with metrics

### `og-image.png`
- Open Graph image for social media sharing
- Recommended: 1200x630px
- Include project title and key visual

## How to Create

### Using Screen Recording

1. Open the simulator in Chrome
2. Set K = 0.5, press Play
3. Record 15 seconds
4. Gradually increase K to 3.0
5. Export as GIF using tools like:
   - [ScreenToGif](https://www.screentogif.com/) (Windows)
   - [Gifski](https://gif.ski/) (macOS)
   - `ffmpeg` (command line)

### Screenshot Tips

- Use a dark browser theme
- Capture at 2x resolution, then resize
- Include visible R(t) transition to show dynamics
- Show the UGFT panel with "Stable" state

## License

All assets in this folder are part of the UGFT Simulator project and are licensed under MIT.
