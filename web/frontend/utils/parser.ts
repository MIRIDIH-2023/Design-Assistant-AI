/**
 * UDOP API의 Output에서 레이아웃 정보를 추출합니다.
 * @param str UDOP API의 Output
 * @returns [x0, y0, x1, y1]의 리스트
 */
export function parseLayout(str: string): number[][] {
  const lists = str.split(/<extra_l_id_\d+>/).filter(Boolean);
  return lists.map((list) =>
    list.match(/<loc_(\d+)>/g)!.map((loc) => parseInt(loc.slice(5, -1)) / 500)
  );
}
