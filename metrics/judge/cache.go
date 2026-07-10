package judge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// Cache wraps a JudgeFunc with an on-disk result cache keyed by prompt hash.
// This is critical for reproducibility: same trajectory + same prompt + same
// thresholds => same score, flattening LLM non-determinism.
type Cache struct {
	dir string
	mu  sync.Mutex
}

// NewCache creates a cache rooted at dir (created if missing).
func NewCache(dir string) (*Cache, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	return &Cache{dir: dir}, nil
}

// Wrap returns a JudgeFunc that consults and populates the cache.
func (c *Cache) Wrap(j JudgeFunc) JudgeFunc {
	return func(ctx context.Context, prompt string) (string, error) {
		key := promptKey(prompt)
		path := filepath.Join(c.dir, key+".json")
		if data, err := os.ReadFile(path); err == nil {
			var cached struct {
				Response string `json:"response"`
			}
			if json.Unmarshal(data, &cached) == nil {
				return cached.Response, nil
			}
		}
		resp, err := j(ctx, prompt)
		if err != nil {
			return "", err
		}
		c.mu.Lock()
		defer c.mu.Unlock()
		payload, _ := json.Marshal(struct {
			Response string `json:"response"`
		}{Response: resp})
		_ = os.WriteFile(path, payload, 0o644)
		return resp, nil
	}
}

func promptKey(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:8])
}
