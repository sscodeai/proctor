// Package llmjudge adapts any OpenAI-compatible chat-completions endpoint to
// a metrics/judge.JudgeFunc. Zero third-party dependencies — plain net/http.
//
// Configuration via env (or explicit options):
//
//	LLM_BASE_URL  e.g. https://api.deepseek.com/v1
//	LLM_API_KEY
//	LLM_MODEL     e.g. deepseek-chat
package llmjudge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/sscodeai/proctor/metrics/judge"
)

// Options configures the OpenAI-compatible client.
type Options struct {
	BaseURL string
	APIKey  string
	Model   string
	Timeout time.Duration // default 60s
	// MaxTokens caps the judge response (default 1024).
	MaxTokens int
	// Temperature (default 0).
	Temperature float64
}

// Client is a thin OpenAI-compatible chat client that exposes a JudgeFunc.
type Client struct {
	opts Options
	http *http.Client
	mu   sync.Mutex
	last judge.Usage
}

// FromEnv builds a Client from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.
func FromEnv() (*Client, error) {
	base := os.Getenv("LLM_BASE_URL")
	key := os.Getenv("LLM_API_KEY")
	model := os.Getenv("LLM_MODEL")
	if base == "" || key == "" || model == "" {
		return nil, fmt.Errorf("set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL")
	}
	return New(Options{BaseURL: base, APIKey: key, Model: model})
}

// New builds a Client from options.
func New(opts Options) (*Client, error) {
	if opts.BaseURL == "" || opts.APIKey == "" || opts.Model == "" {
		return nil, fmt.Errorf("BaseURL, APIKey and Model are required")
	}
	if opts.Timeout == 0 {
		opts.Timeout = 60 * time.Second
	}
	if opts.MaxTokens == 0 {
		opts.MaxTokens = 2048
	}
	return &Client{
		opts: opts,
		http: &http.Client{Timeout: opts.Timeout},
	}, nil
}

// Judge returns a metrics/judge.JudgeFunc bound to this client.
func (c *Client) Judge() judge.JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		return c.chat(ctx, prompt)
	}
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			Reasoning string `json:"reasoning"`
		} `json:"message"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (c *Client) chat(ctx context.Context, prompt string) (string, error) {
	body, _ := json.Marshal(chatRequest{
		Model: c.opts.Model,
		Messages: []chatMessage{
			{Role: "system", Content: "You are an objective evaluation judge. Respond with ONLY the requested JSON in the content field — no reasoning, no markdown, no commentary. Do not put the answer in a reasoning field."},
			{Role: "user", Content: prompt},
		},
		Temperature: c.opts.Temperature,
		MaxTokens:   c.opts.MaxTokens,
	})

	base := strings.TrimSuffix(c.opts.BaseURL, "/")
	url := base + "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.opts.APIKey)

	resp, err := c.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("chat request: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("chat API %d: %s", resp.StatusCode, truncate(string(data), 200))
	}

	var parsed chatResponse
	if err := json.Unmarshal(data, &parsed); err != nil {
		return "", fmt.Errorf("parse chat response: %w", err)
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("chat API error: %s (%s)", parsed.Error.Message, parsed.Error.Type)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("chat API: empty choices")
	}

	if parsed.Usage != nil {
		c.mu.Lock()
		c.last = judge.Usage{
			InputTokens:  parsed.Usage.PromptTokens,
			OutputTokens: parsed.Usage.CompletionTokens,
		}
		c.mu.Unlock()
	}

	content := parsed.Choices[0].Message.Content
	// Some reasoning models (e.g. deepseek-v4-flash via some gateways) put
	// the answer inside `reasoning` and leave content empty. The reasoning
	// field mixes prose with the final JSON — extract just the JSON object.
	if strings.TrimSpace(content) == "" {
		content = extractJSONFromText(parsed.Choices[0].Message.Reasoning)
	}
	return content, nil
}

// extractJSONFromText finds the first JSON OBJECT inside a text blob
// (used to salvage answers from reasoning fields). Arrays are skipped —
// judge responses are always objects, and arrays in reasoning are usually
// conversation noise like "[0] user: ...". Returns the raw text if no
// object is found so the caller's own JSON extraction can fail with context.
func extractJSONFromText(s string) string {
	start := -1
	depth := 0
	inStr := false
	esc := false
	for i, r := range s {
		if inStr {
			if esc {
				esc = false
			} else if r == '\\' {
				esc = true
			} else if r == '"' {
				inStr = false
			}
			continue
		}
		switch r {
		case '"':
			inStr = true
		case '{':
			if start == -1 {
				start = i
			}
			depth++
		case '}':
			depth--
			if depth == 0 && start != -1 {
				return s[start : i+1]
			}
		}
	}
	return s
}

// LastUsage returns the most recent call's usage (for Meter.UsageFn).
func (c *Client) LastUsage() judge.Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.last
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
