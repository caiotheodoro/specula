from specula_model.train_config import SMOKE_MAX_STEPS, dummy_sft_records, trainer_kwargs


def test_smoke_trainer_kwargs_use_4bit_and_capped_steps():
    kwargs = trainer_kwargs(smoke=True)
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_steps"] == SMOKE_MAX_STEPS == 4
    assert kwargs["num_train_epochs"] == 1
    assert kwargs["max_seq_length"] == 256
    assert kwargs["loss_type"] == "nll"


def test_full_trainer_kwargs_keep_4bit_and_honor_epochs():
    kwargs = trainer_kwargs(smoke=False, epochs=3)
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_steps"] == -1
    assert kwargs["num_train_epochs"] == 3
    assert kwargs["max_seq_length"] == 1024
    assert kwargs["loss_type"] == "nll"


def test_dummy_sft_records_are_trl_conversational():
    records = dummy_sft_records()
    assert records
    for record in records:
        turns = record["messages"]
        assert [turn["role"] for turn in turns] == ["user", "assistant"]
        assert "FDA" in turns[0]["content"]
        assert '"verdict"' in turns[1]["content"]
