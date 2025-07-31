import React from "react";

interface LogoProps {
  width?: number | string;
  height?: number | string;
  className?: string;
  style?: React.CSSProperties;
}

const Logo: React.FC<LogoProps> = ({
  width,
  height = 70,
  className = "",
  style = {},
}) => {
  // Don't pass width to SVG if it's "auto" - let CSS handle it
  const svgProps: React.SVGProps<SVGSVGElement> = {
    height,
    viewBox: "0 0 325 70",
    fill: "none",
    xmlns: "http://www.w3.org/2000/svg",
    className,
    style,
  };

  // Only add width prop if it's not "auto"
  if (width !== "auto") {
    svgProps.width =
      width || (typeof height === "number" ? (height * 325) / 70 : 325);
  }

  return (
    <svg {...svgProps}>
      <path
        d="M16.6876 0.280773C18.3924 -0.361066 20.3164 0.120559 21.5178 1.48984L37.1639 19.3231H83.8774L99.5235 1.48984C100.725 0.120559 102.649 -0.361066 104.354 0.280773C106.059 0.922612 107.187 2.55359 107.187 4.37518V43.7501H121.041V52.5H98.4372V15.9954L89.1478 26.5834C88.3171 27.5302 87.1187 28.073 85.8591 28.073H35.1822C33.9226 28.073 32.7242 27.5302 31.8935 26.5834L22.6041 15.9954V52.5H0V43.7501H13.8541V4.37518C13.8541 2.55359 14.9828 0.922612 16.6876 0.280773Z"
        fill="currentColor"
      />
      <path d="M98.4372 61.25H121.041V70H98.4372V61.25Z" fill="currentColor" />
      <path d="M0 61.25H22.6041V70H0V61.25Z" fill="currentColor" />
      <path
        d="M46.6665 46.6667C46.6665 49.8884 44.0549 52.5001 40.8332 52.5001C37.6116 52.5001 34.9999 49.8884 34.9999 46.6667C34.9999 43.4451 37.6116 40.8334 40.8332 40.8334C44.0549 40.8334 46.6665 43.4451 46.6665 46.6667Z"
        fill="currentColor"
      />
      <path
        d="M86.0414 46.6667C86.0414 49.8884 83.4298 52.5001 80.2081 52.5001C76.9865 52.5001 74.3748 49.8884 74.3748 46.6667C74.3748 43.4451 76.9865 40.8334 80.2081 40.8334C83.4298 40.8334 86.0414 43.4451 86.0414 46.6667Z"
        fill="currentColor"
      />
      <path
        d="M166.041 25L146.041 45M146.041 25L166.041 45"
        stroke="currentColor"
        strokeOpacity="0.3"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M245.16 1.45833L257.753 23.3333L231.728 68.5417C231.208 69.4422 230.248 70 229.209 70H205.703C205.183 70 204.685 69.8615 204.249 69.6099C203.813 69.3583 203.445 68.9938 203.184 68.5417L191.431 48.125C190.911 47.2245 190.911 46.1125 191.431 45.2083L216.62 1.45833C216.878 1.00625 217.249 0.64167 217.685 0.390108C218.121 0.138544 218.619 0 219.139 0H242.644C243.684 0 244.644 0.557813 245.164 1.45833H245.16ZM324.078 45.2083L298.889 1.45833C298.631 1.00625 298.261 0.64167 297.824 0.390108C297.388 0.138544 296.89 0 296.37 0H272.865C271.825 0 270.865 0.557813 270.345 1.45833L257.753 23.3333L283.778 68.5417C284.298 69.4422 285.257 70 286.297 70H309.803C310.322 70 310.82 69.8615 311.257 69.6099C311.693 69.3583 312.06 68.9938 312.321 68.5417L324.075 48.125C324.595 47.2245 324.595 46.1125 324.075 45.2083H324.078Z"
        fill="#62DE61"
      />
      <path
        d="M230.892 23.3333H257.756L245.164 1.45833C244.644 0.557813 243.684 0 242.644 0H219.139C218.619 0 218.121 0.138544 217.685 0.390108L230.892 23.3333Z"
        fill="url(#paint0_linear_537_1173)"
      />
      <path
        d="M230.892 23.3333L217.685 0.390108C217.249 0.64167 216.881 1.00625 216.62 1.45833L191.431 45.2083C190.911 46.1125 190.911 47.2208 191.431 48.125L203.184 68.5417C203.442 68.9938 203.813 69.3583 204.249 69.6099L230.888 23.3333L230.892 23.3333Z"
        fill="url(#paint1_linear_537_1173)"
      />
      <path
        d="M257.753 23.3333L230.888 23.3333L204.249 69.6099C204.685 69.8615 205.183 70 205.703 70H229.209C230.248 70 231.208 69.4422 231.728 68.5417L257.753 23.3333Z"
        fill="#09AF58"
      />
      <path
        d="M324.078 48.125C324.336 47.6729 324.467 47.1698 324.467 46.6667H298.053L284.846 69.6099C285.282 69.8615 285.78 70 286.3 70H309.806C310.846 70 311.805 69.4422 312.325 68.5417L324.078 48.125Z"
        fill="#09AF58"
      />
      <path
        d="M272.865 0C272.345 0 271.847 0.138544 271.41 0.390108L298.049 46.6667H324.463C324.463 46.1635 324.333 45.6604 324.075 45.2083L298.889 1.45833C298.369 0.557813 297.41 0 296.37 0H272.861H272.865Z"
        fill="url(#paint2_linear_537_1173)"
      />
      <path
        d="M284.843 69.6098L298.049 46.6667L271.41 0.390108C270.974 0.64167 270.607 1.00625 270.345 1.45833L257.753 23.3333L283.778 68.5417C284.036 68.9938 284.407 69.3583 284.843 69.6098Z"
        fill="url(#paint3_linear_537_1173)"
      />
      <defs>
        <linearGradient
          id="paint0_linear_537_1173"
          x1="290.914"
          y1="87.5"
          x2="234.151"
          y2="-21.3207"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#BFF9B4" />
          <stop offset="1" stopColor="#80EE64" />
        </linearGradient>
        <linearGradient
          id="paint1_linear_537_1173"
          x1="201.536"
          y1="64.1009"
          x2="214.75"
          y2="-11.4523"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#80EE64" />
          <stop offset="0.18" stopColor="#7BEB63" />
          <stop offset="0.36" stopColor="#6FE562" />
          <stop offset="0.55" stopColor="#5ADA60" />
          <stop offset="0.74" stopColor="#3DCA5D" />
          <stop offset="0.93" stopColor="#18B759" />
          <stop offset="1" stopColor="#09AF58" />
        </linearGradient>
        <linearGradient
          id="paint2_linear_537_1173"
          x1="299.983"
          y1="78.4547"
          x2="264.346"
          y2="-24.7324"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#BFF9B4" />
          <stop offset="1" stopColor="#80EE64" />
        </linearGradient>
        <linearGradient
          id="paint3_linear_537_1173"
          x1="294.581"
          y1="63.9735"
          x2="261.192"
          y2="6.30744"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#80EE64" />
          <stop offset="0.18" stopColor="#7BEB63" />
          <stop offset="0.36" stopColor="#6FE562" />
          <stop offset="0.55" stopColor="#5ADA60" />
          <stop offset="0.74" stopColor="#3DCA5D" />
          <stop offset="0.93" stopColor="#18B759" />
          <stop offset="1" stopColor="#09AF58" />
        </linearGradient>
      </defs>
    </svg>
  );
};

export default Logo;
