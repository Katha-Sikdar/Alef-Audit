from alef.objectives import DEFAULT_OBJECTIVE_MAPPINGS, IPIObjective, load_objective_mappings


def test_default_mappings_cover_all_objectives():
    for objective in IPIObjective:
        assert objective in DEFAULT_OBJECTIVE_MAPPINGS


def test_load_objective_mappings_from_bundled_yaml_matches_defaults():
    loaded = load_objective_mappings()
    for objective in IPIObjective:
        assert loaded[objective].realistic_task == DEFAULT_OBJECTIVE_MAPPINGS[objective].realistic_task
        assert loaded[objective].tools_required == DEFAULT_OBJECTIVE_MAPPINGS[objective].tools_required
