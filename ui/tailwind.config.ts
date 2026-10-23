import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#F7F6F3",
        surface: "#FFFFFF",
        subtle: "#F9F9F8",
        mutedBg: "#F1F0EC",
        ink: "#111111",
        bodyInk: "#2F3437",
        mutedInk: "#787774",
        hairline: "#EAEAEA",
        pastelRedBg: "#FDEBEC",
        pastelRedFg: "#9F2F2D",
        pastelBlueBg: "#E1F3FE",
        pastelBlueFg: "#1F6C9F",
        pastelGreenBg: "#EDF3EC",
        pastelGreenFg: "#346538",
        pastelYellowBg: "#FBF3DB",
        pastelYellowFg: "#956400",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "'SF Pro Display'", "'Helvetica Neue'", "sans-serif"],
        serif: ["var(--font-serif)", "'Newsreader'", "Georgia", "serif"],
        mono: ["var(--font-mono)", "'SF Mono'", "monospace"],
      },
      borderRadius: {
        card: "10px",
        btn: "6px",
        kbd: "4px",
      },
    },
  },
  plugins: [],
};

export default config;
