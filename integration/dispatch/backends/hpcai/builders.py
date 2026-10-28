def hpcai_types(*rows: tuple[str, int, float, bool]) -> dict:
    """A `/resource/user/instance/list` answer, one family per (gpu, count, rate, in stock)."""
    return {
        "instanceInfos": [
            {
                "gpuName": gpu,
                "regionInfos": [
                    {
                        "regionName": "us-west-1",
                        "regionId": f"region-{gpu}",
                        "instanceTypeInfos": [
                            {
                                "gpuNum": count,
                                "instanceTypeId": f"type-{gpu}",
                                "stockStatus": "InStock" if stocked else "OutOfStock",
                                "price": [{"chargeMode": "perHour", "price": rate}],
                            }
                        ],
                    }
                ],
            }
            for gpu, count, rate, stocked in rows
        ]
    }
