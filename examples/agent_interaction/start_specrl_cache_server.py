from specrl.suffix_cache import RolloutCacheServer


def main():
    server = RolloutCacheServer("[::]:6378")
    server.initialize()
    print("Rollout cache server listening on [::]:6378", flush=True)
    server.start()
    server.wait()


if __name__ == "__main__":
    main()
