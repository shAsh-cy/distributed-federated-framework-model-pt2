/**
 * The smallest thing that can be called a router.
 *
 * The dashboard has four views held in state and one route that has to be
 * linkable — /story — so this is not worth a routing dependency. Both the
 * path and the hash form are recognised, because the path form needs an SPA
 * fallback on whatever is serving the build and the hash form does not.
 */
import { useCallback, useEffect, useState } from "react";

const CHANGED = "fl:routechange";

export type Route = "dashboard" | "story";

export function readRoute(): Route {
  if (typeof window === "undefined") return "dashboard";
  const { pathname, hash } = window.location;
  return pathname.replace(/\/+$/, "").endsWith("/story") || hash === "#/story"
    ? "story"
    : "dashboard";
}

export function useRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(readRoute);

  useEffect(() => {
    const sync = () => setRoute(readRoute());
    window.addEventListener("popstate", sync);
    window.addEventListener("hashchange", sync);
    window.addEventListener(CHANGED, sync);
    return () => {
      window.removeEventListener("popstate", sync);
      window.removeEventListener("hashchange", sync);
      window.removeEventListener(CHANGED, sync);
    };
  }, []);

  const navigate = useCallback((next: Route) => {
    const base = window.location.pathname.replace(/\/story\/?$/, "").replace(/\/+$/, "");
    const target = next === "story" ? `${base}/story` : `${base}/` || "/";
    try {
      window.history.pushState({}, "", target);
    } catch {
      // file:// and other opaque origins reject pushState; the hash still works.
      window.location.hash = next === "story" ? "#/story" : "";
    }
    window.dispatchEvent(new Event(CHANGED));
  }, []);

  return [route, navigate];
}
