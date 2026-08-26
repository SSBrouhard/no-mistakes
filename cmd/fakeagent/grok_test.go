package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExtractGrokPromptReadsManagedPromptFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(path, []byte("review this branch"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := extractGrokPrompt([]string{"--prompt-file", path, "--output-format", "streaming-messages-json"})
	if err != nil {
		t.Fatalf("extractGrokPrompt() error = %v", err)
	}
	if got != "review this branch" {
		t.Fatalf("prompt = %q", got)
	}
}

func TestRunGrokAcceptsNoContextFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(path, []byte("review"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldStdout })

	status := runGrok([]string{"--no-context-files", "--prompt-file", path, "--output-format", "streaming-messages-json"}, defaultScenario())
	_ = w.Close()
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if status != 0 {
		t.Fatalf("runGrok() status = %d with --no-context-files", status)
	}
	if !strings.Contains(out.String(), `"type":"result"`) {
		t.Fatalf("output missing result after --no-context-files:\n%s", out.String())
	}
}

func TestRunGrokHelpAdvertisesNoContextFiles(t *testing.T) {
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldStdout })

	status := runGrok([]string{"--help"}, defaultScenario())
	_ = w.Close()
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if status != 0 {
		t.Fatalf("runGrok --help status = %d", status)
	}
	if !strings.Contains(out.String(), "--no-context-files") {
		t.Fatalf("fakeagent grok --help must advertise --no-context-files:\n%s", out.String())
	}
}

func TestRunGrokEmitsStreamingMessagesStructuredResult(t *testing.T) {
	path := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(path, []byte("review"), 0o600); err != nil {
		t.Fatal(err)
	}
	oldStdout := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	t.Cleanup(func() { os.Stdout = oldStdout })

	status := runGrok([]string{"--prompt-file", path}, defaultScenario())
	_ = w.Close()
	var out bytes.Buffer
	_, _ = out.ReadFrom(r)
	if status != 0 {
		t.Fatalf("runGrok() status = %d", status)
	}
	for _, want := range []string{`"type":"system"`, `"type":"assistant"`, `"type":"result"`, `"structured_output"`, `"model":"grok-default"`} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %s:\n%s", want, out.String())
		}
	}
}
