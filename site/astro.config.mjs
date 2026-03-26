// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://phillibin.github.io',
  base: '/enough',
  vite: {
    plugins: [tailwindcss()],
  },
});
