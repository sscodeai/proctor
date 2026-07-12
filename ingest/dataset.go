// Package ingest loads evaluation datasets (samples) from JSON/JSONL files
// and converts external agent-trajectory formats into the internal Sample
// model (record-then-evaluate: no runtime instrumentation needed).
package ingest

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"

	"github.com/sscodeai/proctor/trajectory"
)

// LoadSamples reads a dataset file. Supported formats:
//   - .json:  {"samples": [...]} or a bare array of samples
//   - .jsonl: one sample per line
//   - .traj.json: an external trajectory export (OpenAI / LangGraph)
func LoadSamples(path string) ([]trajectory.Sample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}
	return ParseSamples(data, path)
}

// ParseSamples parses dataset bytes, dispatching on file extension.
func ParseSamples(data []byte, path string) ([]trajectory.Sample, error) {
	if len(path) >= 6 && path[len(path)-6:] == ".jsonl" {
		return parseJSONL(data)
	}
	// Try external trajectory formats first (they have distinct envelopes).
	if samples, ok, err := ParseOpenAITrace(data); err != nil {
		return nil, fmt.Errorf("parse OpenAI trace: %w", err)
	} else if ok {
		return samples, nil
	}
	if samples, ok, err := ParseLangGraphTrace(data); err != nil {
		return nil, fmt.Errorf("parse LangGraph trace: %w", err)
	} else if ok {
		return samples, nil
	}
	return parseJSON(data)
}

func parseJSON(data []byte) ([]trajectory.Sample, error) {
	// Try {"samples": [...]} envelope first.
	var env struct {
		Samples []trajectory.Sample `json:"samples"`
	}
	if err := json.Unmarshal(data, &env); err == nil && env.Samples != nil {
		return env.Samples, nil
	}
	// Try bare array.
	var arr []trajectory.Sample
	if err := json.Unmarshal(data, &arr); err != nil {
		return nil, fmt.Errorf("parse samples (envelope or array): %w", err)
	}
	return arr, nil
}

func parseJSONL(data []byte) ([]trajectory.Sample, error) {
	var out []trajectory.Sample
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		var s trajectory.Sample
		if err := json.Unmarshal(line, &s); err != nil {
			return nil, fmt.Errorf("parse jsonl line: %w", err)
		}
		out = append(out, s)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
