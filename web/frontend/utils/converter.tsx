import { Image, Text } from "react-pptx";

export function convertLayoutToComponent(
  layout: number[],
  text: string,
  imgUrl?: string
) {
  if (layout[0] > layout[2]) {
    let temp = layout[0];
    layout[0] = layout[2];
    layout[2] = temp;
  }

  if (layout[1] > layout[3]) {
    let temp = layout[1];
    layout[1] = layout[3];
    layout[3] = temp;
  }

  const width = layout[2] - layout[0];
  const height = layout[3] - layout[1];

  const sm = 15;
  const md = 20;
  const lg = 25;

  let fontSize;

  if (height < 0.05) fontSize = sm;
  else if (height < 0.1) fontSize = md;
  else fontSize = lg;

  return (
    <>
      {imgUrl && (
        // eslint-disable-next-line jsx-a11y/alt-text
        <Image
          src={{
            kind: "path",
            path: imgUrl,
          }}
          style={{
            x: 0,
            y: 0,
            w: "100%",
            h: "100%",
          }}
        />
      )}
      <Text
        style={{
          x: `${Math.round(layout[0] * 100)}%`,
          y: `${Math.round(layout[1] * 100)}%`,
          w: `${Math.round(width * 100)}%`,
          h: `${Math.round(height * 100)}%`,
          fontSize: fontSize,
          align: "center",
          fontFace: "Pretendard",
          bold: true,
        }}
      >
        {text}
      </Text>
    </>
  );
}
