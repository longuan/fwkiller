static main() {
	batch(0);
	auto_wait();
	qexit(1 - load_and_run_plugin("binexport10", 2));
}