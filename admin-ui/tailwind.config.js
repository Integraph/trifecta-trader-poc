/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'Cascadia Code', 'monospace'],
      },
      colors: {
        brand: {
          green:  '#22c55e',
          yellow: '#eab308',
          red:    '#ef4444',
        },
      },
    },
  },
  plugins: [],
};
