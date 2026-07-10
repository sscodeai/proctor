// Package web serves the trajectory-eval visualization UI: a zero-dependency
// Go HTTP server that renders evaluation reports (JSON) as an interactive
// HTML page showing per-sample metrics, attribution (root cause, causal
// chain, evidence), and the trajectory steps.
package web

import (
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/hermes/trajectory-eval/report"
)

//go:embed static/*
var staticFS embed.FS

// Server serves the UI on an address. Report files are loaded from disk
// (or a directory of reports for the V1/V2 diff view).
type Server struct {
	// ReportPath is a JSON report file to display.
	ReportPath string
	// BasePath prefixes all routes (e.g. "/traj-eval" behind a reverse proxy).
	BasePath string
}

// Handler returns the root http.Handler.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	base := s.BasePath
	if base == "" {
		base = "/"
	}
	mux.HandleFunc(base+"api/report", s.handleReport)
	mux.HandleFunc(base+"api/diff", s.handleDiff)
	// Static assets (embedded).
	mux.Handle(base+"static/", http.FileServer(http.FS(staticFS)))
	mux.HandleFunc(base, s.handleIndex)
	return mux
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	html, err := staticFS.ReadFile("static/index.html")
	if err != nil {
		http.Error(w, "index not found", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(html)
}

func (s *Server) handleReport(w http.ResponseWriter, r *http.Request) {
	if s.ReportPath == "" {
		http.Error(w, "no report configured", http.StatusBadRequest)
		return
	}
	rep, err := report.LoadReport(s.ReportPath)
	if err != nil {
		http.Error(w, fmt.Sprintf("load report: %v", err), http.StatusInternalServerError)
		return
	}
	writeJSON(w, rep)
}

func (s *Server) handleDiff(w http.ResponseWriter, r *http.Request) {
	// Diff requires a base report; if not configured, return empty.
	http.Error(w, "diff endpoint not configured", http.StatusNotImplemented)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

// renderStepLine is a helper for the frontend template (kept here for tests).
func renderStepLine(kind, text, tool string) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("<span class=\"step-kind step-%s\">%s</span> ", kind, kind))
	if tool != "" {
		b.WriteString(fmt.Sprintf("<code>%s</code> ", tool))
	}
	b.WriteString(template.HTMLEscapeString(text))
	return b.String()
}

// sortedSampleNames returns sample names in report order.
func sortedSampleNames(rep *report.Report) []string {
	names := make([]string, 0, len(rep.Samples))
	for _, s := range rep.Samples {
		names = append(names, s.Sample)
	}
	sort.Strings(names)
	return names
}

var _ = io.Discard // placeholder to keep io import if unused later
