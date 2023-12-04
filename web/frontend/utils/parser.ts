export function parseLayout(str: string): number[][] {
  const lists = str.split(/<extra_l_id_\d+>/).filter(Boolean);
  return lists.map((list) =>
    list.match(/<loc_(\d+)>/g)!.map((loc) => parseInt(loc.slice(5, -1)) / 500)
  );
}
