package ingest

import (
	"testing"
)

func TestParseJSONEnvelope(t *testing.T) {
	data := []byte(`{"samples": [{"name": "a", "input": "q1"}, {"name": "b", "input": "q2"}]}`)
	samples, err := ParseSamples(data, "golden.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(samples))
	}
	if samples[0].Name != "a" || samples[1].Input != "q2" {
		t.Errorf("unexpected content: %+v", samples)
	}
}

func TestParseJSONArray(t *testing.T) {
	data := []byte(`[{"name": "x"}, {"name": "y"}]`)
	samples, err := ParseSamples(data, "golden.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(samples))
	}
}

func TestParseJSONL(t *testing.T) {
	data := []byte("{\"name\": \"a\"}\n# comment\n{\"name\": \"b\"}\n")
	samples, err := ParseSamples(data, "golden.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != 2 {
		t.Fatalf("expected 2 samples (comment skipped), got %d", len(samples))
	}
}

func TestParseInvalid(t *testing.T) {
	_, err := ParseSamples([]byte(`{not json`), "golden.json")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}
