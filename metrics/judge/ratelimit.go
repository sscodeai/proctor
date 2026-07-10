package judge

import (
	"context"
	"sync"
	"time"
)

// RateLimit wraps a JudgeFunc with a token-bucket rate limiter.
func RateLimit(j JudgeFunc, rps float64, burst int) JudgeFunc {
	if rps <= 0 {
		return j
	}
	var mu sync.Mutex
	tokens := float64(burst)
	last := time.Now()
	return func(ctx context.Context, prompt string) (string, error) {
		mu.Lock()
		now := time.Now()
		elapsed := now.Sub(last).Seconds()
		tokens += elapsed * rps
		if tokens > float64(burst) {
			tokens = float64(burst)
		}
		last = now
		if tokens < 1 {
			// Wait until a token is available.
			wait := time.Duration((1 - tokens) / rps * float64(time.Second))
			mu.Unlock()
			select {
			case <-time.After(wait):
			case <-ctx.Done():
				return "", ctx.Err()
			}
			mu.Lock()
			tokens = 0
			last = time.Now()
		} else {
			tokens--
		}
		mu.Unlock()
		return j(ctx, prompt)
	}
}
