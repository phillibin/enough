import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { marked } from "marked";

const SITE_ROOT = path.resolve(process.cwd());
const DRAFTS_DIR = path.resolve(SITE_ROOT, "../writing/drafts");
const SCORES_PATH = path.resolve(SITE_ROOT, "src/data/scores.json");

interface SentenceScore {
  text: string;
  score: number;
}

interface SectionScores {
  overall: number;
  class: string;
  sentences: SentenceScore[];
}

export interface Draft {
  slug: string;
  filename: string;
  title: string;
  status: string;
  order: number;
  html: string;
  raw: string;
  scores?: SectionScores;
}

function loadScores(): Record<string, SectionScores> {
  try {
    const raw = fs.readFileSync(SCORES_PATH, "utf-8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Inject score markers into raw markdown text BEFORE rendering.
 * Uses HTML spans which marked will pass through untouched.
 */
function injectScoresIntoMarkdown(
  markdown: string,
  sentences: SentenceScore[],
): string {
  let result = markdown;

  // Process sentences longest-first to avoid partial matches
  const sorted = [...sentences].sort((a, b) => b.text.length - a.text.length);

  for (const { text, score } of sorted) {
    // The GPTZero sentence text is plain text (no markdown formatting).
    // We need to find it in markdown where it might contain *emphasis* markers.
    // Strategy: build a regex that allows optional * or _ around words.

    // First try an exact match (works for most sentences)
    const exactIdx = result.indexOf(text);
    if (exactIdx !== -1) {
      // Make sure we're not already inside a slop-sentence span.
      // Check if there's an unclosed <span class="slop-sentence" before this point.
      const before = result.slice(0, exactIdx);
      const lastOpen = before.lastIndexOf('<span class="slop-sentence"');
      const lastClose = before.lastIndexOf("</span>");
      const insideSpan = lastOpen !== -1 && lastOpen > lastClose;
      if (!insideSpan) {
        const span = `<span class="slop-sentence" data-slop="${score.toFixed(3)}">${text}</span>`;
        result = result.slice(0, exactIdx) + span + result.slice(exactIdx + text.length);
        continue;
      }
    }

    // If exact match fails, try matching with markdown emphasis stripped.
    // Build a regex from the plain text that tolerates *...* or _..._ wrappers.
    const words = text.split(/\s+/);
    if (words.length < 2) continue; // skip single-word fragments for fuzzy match

    // Create a pattern that matches each word with optional surrounding * or _
    const pattern = words
      .map((w) => `\\*{0,2}_?${escapeRegex(w)}_?\\*{0,2}`)
      .join("\\s+");

    const regex = new RegExp(pattern);
    const match = result.match(regex);
    if (match && match.index !== undefined) {
      const before = result.slice(0, match.index);
      const lastOpen = before.lastIndexOf('<span class="slop-sentence"');
      const lastClose = before.lastIndexOf("</span>");
      const insideSpan = lastOpen !== -1 && lastOpen > lastClose;
      if (!insideSpan) {
        const original = match[0];
        const span = `<span class="slop-sentence" data-slop="${score.toFixed(3)}">${original}</span>`;
        result =
          result.slice(0, match.index) +
          span +
          result.slice(match.index + original.length);
      }
    }
  }

  return result;
}

function parseDraft(
  filename: string,
  content: string,
  scores: Record<string, SectionScores>,
): Draft {
  const { data: frontmatter, content: body } = matter(content);

  let title = frontmatter.title || "";
  let status = frontmatter.status || "";

  const lines = body.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("# ")) {
      const headingMatch = line.match(/^#\s+(.+?)(?:\s*\[(.+?)\])?\s*$/);
      if (headingMatch) {
        if (!title) title = headingMatch[1].trim();
        if (!status && headingMatch[2]) status = headingMatch[2].trim();
      }
      lines[i] = "";
      continue;
    }
    if (line.startsWith("# STATUS:") || line.startsWith("// STATUS:")) {
      lines[i] = "";
      continue;
    }
  }
  let cleanBody = lines.join("\n").replace(/^\n+/, "");

  const orderMatch = filename.match(/^(\d+)/);
  const order = orderMatch ? parseInt(orderMatch[1], 10) : 99;
  const slug = filename.replace(/\.md$/, "");

  // Inject slop scores into markdown before rendering
  const sectionScores = scores[slug];
  if (sectionScores?.sentences) {
    cleanBody = injectScoresIntoMarkdown(cleanBody, sectionScores.sentences);
  }

  const html = marked.parse(cleanBody) as string;

  return {
    slug,
    filename,
    title,
    status,
    order,
    html,
    raw: cleanBody,
    scores: sectionScores,
  };
}

export function getAllDrafts(): Draft[] {
  const scores = loadScores();
  const files = fs.readdirSync(DRAFTS_DIR).filter((f) => {
    if (!f.endsWith(".md")) return false;
    if (f.startsWith("00_")) return false;
    if (f.startsWith("three_axis")) return false;
    return true;
  });

  return files
    .map((f) =>
      parseDraft(f, fs.readFileSync(path.join(DRAFTS_DIR, f), "utf-8"), scores),
    )
    .sort((a, b) => a.order - b.order);
}

export function getFullText(): Draft[] {
  return getAllDrafts();
}

/** Get only polished sections (01, 02). */
export function getPolishedDrafts(): Draft[] {
  return getAllDrafts().filter((d) => d.order <= 2);
}
